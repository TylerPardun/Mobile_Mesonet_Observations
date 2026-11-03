#!/usr/bin/env python3
"""Audit archived CH2 inputs, reproduce Barnes grids, and quantify uncertainty.

Run --inventory on a login node; run the full calculation through Slurm.
Original inputs and manuscript figures are never overwritten. The return
archive contains compact grids, diagnostics, provenance and candidate figures.
It is a review product, not automatic certification of resubmission readiness.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
from barnes_uncertainty import barnes_jackknife, within_sigma, jackknife_interval, fdr_adjust


class SavedObject:
    def __init__(self,*args,**kwargs): pass


class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith(('sharppy.','sharptab.')):
            return SavedObject
        return super().find_class(module,name)


def load(path):
    with Path(path).open('rb') as f:return Reader(f).load()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def locate(root,name,is_dir=False):
    found=[]
    for base,dirs,files in os.walk(root):
        rel=Path(base).relative_to(root)
        dirs[:]=[d for d in dirs if not d.startswith('.') and d not in
                  {'nexrad','nexrad_grids','mobile_mesonet_raw','node_modules','results','elevation'}]
        if len(rel.parts)>=3:dirs[:]=[]
        if name in (dirs if is_dir else files):found.append(Path(base)/name)
    if len(found)!=1:
        raise ValueError(f'Expected one {name} under {root}; found {found}. Supply its explicit path.')
    return found[0]


def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return str(x)


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=json_default)+'\n')


def metadata_values(obj,key):
    values=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k==key and np.isscalar(v):values.append(v)
            elif isinstance(v,(dict,list,tuple)):values.extend(metadata_values(v,key))
    elif isinstance(obj,(list,tuple)):
        for v in obj:values.extend(metadata_values(v,key))
    return values


def archived_field(path):
    a=load(path)
    needed={'var_barnes','Xg','Yg'}
    if not isinstance(a,dict) or not needed.issubset(a):
        raise ValueError(f'Unrecognized Barnes archive structure: {path}')
    return a


def prepare(entry,mode,variable,cases,days,X,Y,config):
    x,y,z=[np.asarray(entry[k],float).ravel() for k in ('rdx','rdy',variable)]
    if not (len(x)==len(y)==len(z)):
        raise ValueError('Coordinate/value length mismatch')
    if 'case_label' in entry:
        labels=np.asarray(entry['case_label']).astype(str).ravel()
    else:
        ids=np.asarray(entry['caseID'],float).ravel()
        if not np.isfinite(ids).all() or not np.all(ids==np.floor(ids)):
            raise ValueError('Invalid numeric case IDs')
        labels=np.asarray([f'{mode}{int(v)+1}' for v in ids])
    if len(labels)!=len(z):raise ValueError('Case/value length mismatch')
    unknown=set(labels)-set(cases.index)
    if unknown:raise ValueError(f'Unknown cases: {unknown}')
    if any(not label.startswith(mode) for label in labels):raise ValueError('Outcome/case mismatch')
    # Embedded t0 must agree with the supplied case catalogue before day clustering.
    if 't0' in entry:
        times=np.asarray(entry['t0']).astype(str).ravel()
        for label in np.unique(labels):
            observed=set(times[labels==label]);expected=cases.loc[label,'t0'].strftime('%Y%m%d%H%M')
            if observed!={expected}:raise ValueError(f'{label}: cache t0 {observed} differs from catalogue {expected}')
    b=config['prebuffer_km']
    use=(x>=X.min()-b)&(x<=X.max()+b)&(y>=Y.min()-b)&(y<=Y.max()+b)
    indices=np.flatnonzero(use)
    indices=indices[within_sigma(z[indices],config['outlier_sigma'])]
    indices=indices[np.isfinite(z[indices]) & np.isfinite(x[indices]) & np.isfinite(y[indices])]
    if len(indices)<4:raise ValueError('Insufficient finite points after fixed QC')
    if variable=='qv' and np.nanmax(abs(z[indices]))>.2:
        raise ValueError('qv appears not to be in native kg/kg; check input units')
    selected=labels[indices]
    codes=np.asarray([days.index(cases.loc[c,'day']) for c in selected],int)
    return x[indices],y[indices],z[indices],codes,selected


def copy_provenance(root,out,manifest):
    dest=out/'provenance';dest.mkdir()
    names=['case_base_states.csv','case_base_states_original.csv','transect_times.xlsx',
           'mobile_mesonet_analysis.ipynb','build_base_state_fromRAP_RUC.py',
           'barnes_notebook_helpers.py']
    for name in names:
        try:p=locate(root,name)
        except ValueError as exc:
            manifest['warnings'].append(str(exc));continue
        if p.stat().st_size<20*1024*1024:
            shutil.copy2(p,dest/name)
            manifest['inputs'][str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}


def run(args,manifest):
    root=args.root.resolve();out=args.output.resolve()
    config=json.loads(args.config.read_text())
    cache=args.pickle or locate(root,'monte_carlo_stuff.pkl')
    barnes=args.barnes or locate(root,'barnes_precomputed',is_dir=True)
    transects=args.transects or locate(root,'transect_times.xlsx')
    paths=[cache,transects]
    for v in config['variables']:
        for m in ('T','N'):
            for off in config['offsets_min']:
                p=barnes/v/f'{m}_{off:+03d}.pkl'
                if not p.exists():raise FileNotFoundError(p)
                paths.append(p)
    meta_files=[barnes/n for n in ('metadata.json','metadata.pkl') if (barnes/n).exists()]
    chosen_metadata={}
    for path in meta_files:
        meta=json.loads(path.read_text()) if path.suffix=='.json' else load(path)
        write_json(out/('archive_metadata_'+path.suffix.lstrip('.')+'.json'),meta)
        for key in ('kappa','gamma','min_neighbors'):
            values=set(float(v) for v in metadata_values(meta,key))
            if len(values)>1:raise ValueError(f'Conflicting {key} values in {path}: {values}')
            if values:
                value=next(iter(values))
                if key in chosen_metadata and chosen_metadata[key]!=value:
                    raise ValueError(f'JSON and pickle product metadata disagree on {key}')
                config[key]=value;chosen_metadata[key]=value
        paths.append(path)
    config['min_neighbors']=int(config['min_neighbors'])
    if not meta_files:manifest['warnings'].append('Barnes metadata absent: configured parameters require provenance review even if grids reproduce.')
    if set(chosen_metadata)!={'kappa','gamma','min_neighbors'}:
        manifest['warnings'].append('Not all Barnes parameters were supplied by archive metadata; baseline settings are recorded explicitly for review.')
    if config['fdr_primary']!='BY':raise ValueError('This protocol specifies BY as the primary FDR procedure')
    manifest['configuration']=config
    manifest['inventory']=[{'path':str(p),'bytes':p.stat().st_size} for p in paths]
    write_json(out/'inventory.json',manifest)
    if args.inventory:
        manifest['status']='inventory_only';return
    # Avoid an unexpectedly large object load on a 28-GB allocation. The full
    # mobile_mesonet_data_tsc_5_min.pkl is deliberately never opened.
    if cache.stat().st_size>8*1024**3:
        raise MemoryError('Composite cache exceeds 8 GB. Return inventory.json for a targeted extractor; do not load the full synthesis.')
    print('Hashing inputs and loading the composite cache...',flush=True)
    for p in paths:manifest['inputs'][str(p)]={'sha256':digest(p),'bytes':p.stat().st_size}
    copy_provenance(root,out,manifest)
    payload=load(cache)
    if not isinstance(payload,(tuple,list)) or not isinstance(payload[0],dict):
        raise ValueError('Expected [mesonet_data, sounding_data, ...] cache')
    mesonet=payload[0];del payload;gc.collect()
    # Export small provenance summaries before any reproduction or t0 check can
    # fail, so a failed remote run is still useful without transferring raw data.
    cache_rows=[]
    for mode in ('T','N'):
        for offset in config['offsets_min']:
            entry=mesonet[mode][offset]
            labels=np.asarray(entry['case_label']).astype(str) if 'case_label' in entry else np.asarray([f'{mode}{int(v)+1}' for v in entry['caseID']])
            for label in np.unique(labels):
                selected=labels==label
                row=dict(mode=mode,case_id=label,offset_min=offset,n_obs=int(selected.sum()))
                if 't0' in entry:row['embedded_t0']=';'.join(sorted(set(np.asarray(entry['t0']).astype(str)[selected])))
                for key in ('base_temp','base_rh','base_pres','base_pres_corr','z_ref_m','u_storm','v_storm'):
                    if key in entry:
                        a=np.asarray(entry[key],float)[selected];a=a[np.isfinite(a)]
                        if len(a):row[key+'_min']=float(a.min());row[key+'_max']=float(a.max())
                for plain,perturbed in (('gru','pgru'),('grv','pgrv')):
                    if plain in entry and perturbed in entry:
                        a=(np.asarray(entry[plain],float)-np.asarray(entry[perturbed],float))[selected]
                        a=a[np.isfinite(a)]
                        if len(a):row['implied_base_'+plain+'_min']=float(a.min());row['implied_base_'+plain+'_max']=float(a.max())
                cache_rows.append(row)
    pd.DataFrame(cache_rows).to_csv(out/'cache_case_inventory.csv',index=False)
    tr=pd.read_excel(transects,sheet_name='transect_times')
    tr['t0']=pd.to_datetime(tr.Tstart,format='%Y%m%d-%H%M')
    tr['day']=(tr.t0-pd.Timedelta(hours=config['storm_day_start_utc_hour'])).dt.strftime('%Y-%m-%d')
    if tr.CaseID.duplicated().any():raise ValueError('Duplicate cases in catalogue')
    cases=tr.set_index('CaseID');days=sorted(cases.day.unique());G=len(days)
    if G<3:raise ValueError('Too few storm days')
    case_counts=tr.CaseID.str[0].value_counts().to_dict()
    manifest['sampling']={'cases':case_counts,'joint_storm_days':G,'day_definition':'1200–1200 UTC'}
    if not args.allow_fixture and case_counts!={'N':57,'T':52}:
        raise ValueError(f'Expected submitted 52 T / 57 N case catalogue; found {case_counts}')
    products=[];reproduction=[];base_rows=[]
    for variable in config['variables']:
        for offset in config['offsets_min']:
            groups={};saved={};common_grid=None
            for mode in ('T','N'):
                print(f'{variable} {offset:+d} min {mode}: full and {G} delete-day analyses',flush=True)
                archive_path=barnes/variable/f'{mode}_{offset:+03d}.pkl'
                archive=archived_field(archive_path)
                X,Y=[np.asarray(archive[k],float) for k in ('Xg','Yg')]
                if X.shape!=Y.shape or not np.allclose(X,np.tile(X[0,:],(len(X),1))) or not np.allclose(Y,np.tile(Y[:,0,None],(1,X.shape[1]))):
                    raise ValueError('Archive grid must be rectangular, Cartesian, increasing, and in km')
                if np.any(np.diff(X[0,:])<=0) or np.any(np.diff(Y[:,0])<=0):raise ValueError('Non-increasing grid')
                if common_grid is None:common_grid=(X.copy(),Y.copy())
                elif not (np.array_equal(X,common_grid[0]) and np.array_equal(Y,common_grid[1])):
                    raise ValueError('Tornadic and nontornadic coordinates differ')
                entry=mesonet[mode][offset]
                x,y,z,code,labels=prepare(entry,mode,variable,cases,days,X,Y,config)
                result=barnes_jackknife(x,y,z,code,X,Y,G,kappa=config['kappa'],gamma=config['gamma'],
                    min_neighbors=config['min_neighbors'],chunk=config['kernel_chunk'])
                expected=np.asarray(np.ma.filled(archive['var_barnes'],np.nan),float)
                computed=result['mean']
                same_mask=np.array_equal(np.isfinite(computed),np.isfinite(expected))
                error=float(np.nanmax(abs(computed-expected)))
                ok=same_mask and np.allclose(computed,expected,atol=config['reproduction_atol'],rtol=config['reproduction_rtol'],equal_nan=True)
                reproduction.append(dict(variable=variable,offset=offset,mode=mode,max_abs_error_native=error,same_finite_mask=same_mask,passed=ok,n_obs=len(z),n_cases=len(set(labels))))
                pd.DataFrame(reproduction).to_csv(out/'grid_reproduction.csv',index=False)
                if not ok:
                    np.savez_compressed(out/f'mismatch_{variable}_{mode}_{offset:+03d}.npz',X=X,Y=Y,computed=computed,archived=expected)
                    raise ValueError('Archived grids do not reproduce. Return the diagnostics; no significance maps have been certified.')
                low,high,se=jackknife_interval(computed,result['leave_one_out'],np.maximum(result['n_days']-1,1))
                eligible=(result['effective_days']>=config['minimum_effective_days'])&np.isfinite(result['leave_one_out']).all(axis=0)
                scale=1000. if variable=='qv' else 1.
                for key,value in {'mean':computed,'ci_low':low,'ci_high':high,'se':se}.items():
                    saved[f'{mode}_{key}']=value*scale
                saved[f'{mode}_eligible']=eligible
                saved[f'{mode}_effective_days']=result['effective_days']
                saved[f'{mode}_n_days']=result['n_days']
                saved[f'{mode}_n_cases']=np.asarray(len(set(labels)))
                for key in ('var_u_barnes','var_v_barnes','var_std'):
                    if key in archive:saved[f'{mode}_{key}']=np.asarray(np.ma.filled(archive[key],np.nan))
                # Copy the independently saved reflectivity context if available.
                refname=f'ref_composite_{mode.lower()}_height_3p00km_{offset}min.pkl'
                candidates=[root/'data/dbz_composites'/refname,root/'dbz_composites'/refname]
                refs=[p for p in candidates if p.exists()]
                if len(refs)==1:
                    rx,ry,dbz=load(refs[0])
                    saved[mode+'_ref_x']=np.asarray(rx)/1000.
                    saved[mode+'_ref_y']=np.asarray(ry)/1000.
                    saved[mode+'_reflectivity']=np.asarray(dbz)
                    manifest['inputs'][str(refs[0])]={'sha256':digest(refs[0]),'bytes':refs[0].stat().st_size}
                elif variable==config['variables'][0]:
                    manifest['warnings'].append(f'Reflectivity context missing or ambiguous: {refname}')
                groups[mode]=result
                if variable==config['variables'][0]:
                    all_labels=np.asarray(entry.get('case_label',[f'{mode}{int(i)+1}' for i in entry['caseID']])).astype(str)
                    for label in np.unique(all_labels):
                        row=dict(case_id=label,offset_min=offset)
                        for key in ('base_temp','base_rh','base_pres','base_pres_corr','z_ref_m'):
                            if key in entry:
                                a=np.asarray(entry[key],float)[all_labels==label]
                                row[key+'_min']=float(np.nanmin(a));row[key+'_max']=float(np.nanmax(a))
                        base_rows.append(row)
            a,b=groups['T'],groups['N']
            if a['mean'].shape!=b['mean'].shape:raise ValueError('Group grids differ')
            difference=a['mean']-b['mean']
            deleted_difference=a['leave_one_out']-b['leave_one_out']
            df=np.maximum(np.minimum(a['n_days'],b['n_days'])-1,1)
            low,high,se=jackknife_interval(difference,deleted_difference,df)
            eligible=saved['T_eligible']&saved['N_eligible']&(se>0)
            p=np.full(X.shape,np.nan)
            p[eligible]=2*student_t.sf(abs(difference[eligible]/se[eligible]),df[eligible])
            saved.update(X=X,Y=Y,difference=difference*scale,difference_ci_low=low*scale,
                         difference_ci_high=high*scale,p_value=p,test_eligible=eligible)
            products.append((variable,offset,saved))
            del groups;gc.collect()
    pd.DataFrame(base_rows).to_csv(out/'embedded_base_states.csv',index=False)
    family=np.concatenate([a['p_value'].ravel() for _,_,a in products])
    q_bh=fdr_adjust(family);q_by=fdr_adjust(family,True)
    position=0;summary=[]
    for variable,offset,a in products:
        size=a['p_value'].size
        a['q_BH']=q_bh[position:position+size].reshape(a['p_value'].shape)
        a['q_BY']=q_by[position:position+size].reshape(a['p_value'].shape)
        a['significant']=a['q_BY']<=config['fdr_q'];position+=size
        np.savez_compressed(out/f'{variable}_{offset:+03d}.npz',**a)
        summary.append(dict(variable=variable,offset_min=offset,eligible=int(np.isfinite(a['p_value']).sum()),
            significant_BY=int(a['significant'].sum()),significant_BH=int((a['q_BH']<=config['fdr_q']).sum())))
    pd.DataFrame(summary).to_csv(out/'significance_summary.csv',index=False)
    manifest['inference']={'estimator':'pooled-observation two-pass Barnes, unchanged',
        'uncertainty':'delete-one-storm-day jackknife; full two-pass calculation for every deletion',
        'interval':'pointwise t interval; df = supported storm days minus one',
        'difference_df':'smaller of group supported-day counts minus one',
        'hypothesis':'two-sided zero composite difference using jackknife SE and t approximation',
        'dependence':'same joint day omitted from both groups',
        'primary_multiplicity':'Benjamini–Yekutieli at q=.05 across every eligible grid cell in all 20 scalar/time maps',
        'sensitivity':'Benjamini–Hochberg results also returned; not the primary stippling',
        'conditional_on':'original QC mask, event alignment, base states, Barnes parameters, targeted sample',
        'limits':'Sampling uncertainty only; no instrument/base-state/registration uncertainty propagation. Pointwise CIs are not simultaneous. Small effective-day support is masked.'}
    from plot_results import plot_all
    plot_all(out,config)
    manifest['status']='analysis_complete_pending_scientific_review'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/condo/swat_z6/tpardun/phd_stuff'))
    parser.add_argument('--pickle',type=Path)
    parser.add_argument('--barnes',type=Path)
    parser.add_argument('--transects',type=Path)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--allow-fixture',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'started','utc':datetime.now(timezone.utc).isoformat(),
        'inputs':{},'warnings':[],'software':{'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'pandas':pd.__version__},'command':sys.argv,
        'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    code=0
    try:run(args,manifest)
    except Exception as exc:
        manifest['status']='needs_review';manifest['error']=str(exc)
        (args.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc();code=1
    finally:
        write_json(args.output/'run_manifest.json',manifest)
        bundle=args.output.with_suffix('.tar.gz')
        with tarfile.open(bundle,'w:gz') as archive:archive.add(args.output,arcname=args.output.name)
        print(f'Return archive: {bundle}',flush=True)
    sys.exit(code)


if __name__=='__main__':main()
