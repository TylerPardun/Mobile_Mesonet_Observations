"""Render candidate composite figures from compact, audited output grids."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE={
    'pgru':(20,10,r"GR-$u'$ (m s$^{-1}$)",[2,4,6]),
    'thv':(8,4,r"$\theta_v'$ (K)",[1,2,3]),
    'the':(20,10,r"$\theta_e'$ (K)",[2,4,6]),
    'qv':(5,3,r"$q_v'$ (g kg$^{-1}$)",[.5,1,1.5]),
}


def plot_all(out,config):
    out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for var in config['variables']:
        limit,diff_limit,label,ci_levels=STYLE[var]
        offsets=config['offsets_min'];n=len(offsets)
        fig,axes=plt.subplots(3,n,figsize=(3*n,9),squeeze=False,layout='constrained',sharex=True,sharey=True)
        figci,ci_axes=plt.subplots(2,n,figsize=(3*n,6),squeeze=False,layout='constrained',sharex=True,sharey=True)
        for col,offset in enumerate(offsets):
            a=np.load(out/f'{var}_{offset:+03d}.npz');X,Y=a['X'],a['Y']
            for row,mode in enumerate(('T','N')):
                ax=axes[row,col]
                im=ax.pcolormesh(X,Y,a[mode+'_mean'],vmin=-limit,vmax=limit,cmap='RdBu_r',shading='auto',rasterized=True)
                half=(a[mode+'_ci_high']-a[mode+'_ci_low'])/2
                half=np.where(a[mode+'_eligible'],half,np.nan)
                available=[v for v in ci_levels if np.isfinite(half).any() and np.nanmin(half)<v<np.nanmax(half)]
                if available:
                    cs=ax.contour(X,Y,half,levels=available,colors='.25',linewidths=.6)
                    ax.clabel(cs,fontsize=7,fmt='%g')
                if mode+'_var_u_barnes' in a and mode+'_var_v_barnes' in a:
                    step=max(1,len(X)//9)
                    vectors=ax.quiver(X[::step,::step],Y[::step,::step],a[mode+'_var_u_barnes'][::step,::step],
                                      a[mode+'_var_v_barnes'][::step,::step],scale=130,width=.004,color='.15')
                    if col==0:
                        ax.quiverkey(vectors,.75,.08,10,'10 m/s',coordinates='axes',labelpos='N',fontproperties={'size':7})
                if mode+'_reflectivity' in a:
                    dbz=np.asarray(a[mode+'_reflectivity'],float)
                    if dbz.ndim==2:
                        ax.contour(a[mode+'_ref_x'],a[mode+'_ref_y'],dbz,
                                   levels=[35,40,45,50],colors='k',linewidths=1.)
                ax.set_title(f'({chr(97+row*n+col)}) {mode}, {offset:+d} min; n={int(a[mode+"_n_cases"])}')
                cim=ci_axes[row,col].pcolormesh(X,Y,half,vmin=0,vmax=max(ci_levels)*1.5,cmap='viridis',shading='auto',rasterized=True)
                ci_axes[row,col].set_title(f'{mode}, {offset:+d} min')
            ax=axes[2,col]
            dim=ax.pcolormesh(X,Y,a['difference'],vmin=-diff_limit,vmax=diff_limit,cmap='RdBu_r',shading='auto',rasterized=True)
            significant=a['significant'];step=max(1,len(X)//35)
            ss=significant[::step,::step];xx=X[::step,::step];yy=Y[::step,::step]
            ax.scatter(xx[ss],yy[ss],s=2,c='k',linewidths=0)
            if np.any(~a['test_eligible']):
                ax.contourf(X,Y,np.where(a['test_eligible'],np.nan,1.),levels=[.5,1.5],colors=['.8'],alpha=.35)
            ax.set_title(f'({chr(97+2*n+col)}) T − N; {int(significant.sum())} FDR cells')
            for row in range(3):
                axes[row,col].set_aspect('equal');axes[row,col].axhline(0,c='.5',lw=.5,ls='--');axes[row,col].axvline(0,c='.5',lw=.5,ls='--')
            for row in range(2):ci_axes[row,col].set_aspect('equal')
        for row in range(3):axes[row,0].set_ylabel('Cross-motion distance (km)')
        for ax in axes[-1]:ax.set_xlabel('Along-motion distance (km)')
        for row in range(2):ci_axes[row,0].set_ylabel('Cross-motion distance (km)')
        for ax in ci_axes[-1]:ax.set_xlabel('Along-motion distance (km)')
        fig.colorbar(im,ax=axes[:2,:],shrink=.6,label=label)
        fig.colorbar(dim,ax=axes[2,:],shrink=.6,label='T − N: '+label)
        fig.suptitle(label+' | contours: 95% CI half-width; stippling: global BY FDR q ≤ 0.05\n'
                     'Gray overlay: insufficient effective storm-day support',fontsize=12)
        figci.colorbar(cim,ax=ci_axes,shrink=.7,label='95% CI half-width: '+label)
        figci.suptitle('Pointwise sampling uncertainty, conditional on fixed QC and analysis settings',fontsize=12)
        fig.savefig(figdir/f'{var}_barnes_mwr.png',dpi=300)
        figci.savefig(figdir/f'{var}_confidence_width.png',dpi=250)
        plt.close(fig);plt.close(figci)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args()
    plot_all(args.results,json.loads((args.results/'run_manifest.json').read_text())['configuration'])
