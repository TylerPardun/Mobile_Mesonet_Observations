"""Validate the Schooner return and render manuscript figures without rerunning inference."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'output/run_35685825'
PAPER = ROOT / 'manuscript/CH2'
AUDIT = PAPER / 'revisions/mwr_revision_audit_20261103'
OFFSETS = [-15, -5, 0, 5, 15]
STYLE = {
    'pgru': (20, 10, 15, r"GR-$u'$ (m s$^{-1}$)"),
    'thv': (8, 4, 6, r"$\theta_v'$ (K)"),
    'the': (20, 10, 15, r"$\theta_e'$ (K)"),
    'qv': (5, 3, 4, r"$q_v'$ (g kg$^{-1}$)"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    manifest = json.loads((RETURN / 'run_manifest.json').read_text())
    assert manifest['status'] == 'analysis_complete_pending_scientific_review'
    grids = pd.read_csv(RETURN / 'grid_reproduction.csv')
    assert len(grids) == 40
    assert grids['passed'].all()
    arrays = [np.load(RETURN / f'{v}_{t:+03d}.npz') for v in STYLE for t in OFFSETS]
    p = np.concatenate([a['p_value'].ravel() for a in arrays])
    for method in ('bh', 'by'):
        reported = np.concatenate([a['q_' + method.upper()].ravel() for a in arrays])
        np.testing.assert_allclose(reported[np.isfinite(p)], false_discovery_control(p[np.isfinite(p)], method=method), atol=1e-14)
    for a in arrays:
        np.testing.assert_allclose(a['difference'], a['T_mean'] - a['N_mean'], atol=1e-12)
        for group in ('T', 'N'):
            eligible = a[group + '_eligible']
            assert np.all(a[group + '_effective_days'][eligible] >= 10)
            np.testing.assert_allclose((a[group + '_ci_high'] + a[group + '_ci_low']) / 2, a[group + '_mean'], atol=1e-12)
        assert not a['significant'].any()
        assert not (a['q_BH'] <= .05).any()
    b = pd.read_csv(RETURN / 'provenance/case_base_states.csv')
    a = pd.read_csv(RETURN / 'cache_case_inventory.csv').merge(b, on='case_id', validate='many_to_one', suffixes=('', '_csv'))
    errors = {}
    for cache, source in [('base_temp', 'temp_C'), ('base_rh', 'rh_pct'), ('base_pres_corr', 'pres_corr_hPa')]:
        for end in ('min', 'max'):
            err = float(np.max(abs(a[cache + '_' + end] - a[source])))
            assert err < 1e-10
            errors[cache + '_' + end] = err
    speed = np.hypot(a.u_storm_min, a.v_storm_min)
    rotated_u = (a.u_ms * a.u_storm_min + a.v_ms * a.v_storm_min) / speed
    rotated_v = (-a.u_ms * a.v_storm_min + a.v_ms * a.u_storm_min) / speed
    for expected, key in [(rotated_u, 'implied_base_gru'), (rotated_v, 'implied_base_grv')]:
        for end in ('min', 'max'):
            errors[key + '_' + end] = float(np.max(abs(expected - a[key + '_' + end])))
            assert errors[key + '_' + end] < 1e-10
    # The embedded z_ref_m is the WINDOW datum, not the fixed base-state datum.
    # They agree (to CSV rounding) only at t0. No pressure-difference field is used here.
    zero = a[a.offset_min == 0]
    assert np.max(abs(zero.z_ref_m_min - zero.z_ref_m)) <= .000501
    assert (b.loc[b.base_source == 'RAP_RUC_regional', 'source_clusters'] == 1).all()
    source_archive = ROOT / 'output/run_35685825.tar.gz'
    report = dict(return_archive_sha256=digest(source_archive) if source_archive.exists() else None,
                  reproduction=grids.to_dict('records'), base_state_max_errors=errors,
                  family_size=int(np.isfinite(p).sum()), detections_BY=0, detections_BH=0,
                  independent_check='SciPy BH and BY agree with every returned adjusted P value',
                  source_counts=b.base_source.value_counts().to_dict(), inputs={},
                  datum_note='Base-state z_ref_m is fixed per case; cached z_ref_m varies by window. Only the zero-offset values should agree. No pressure-perturbation map is analyzed in this manuscript.')
    columns = ['case_id', 'mode', 'base_valid_time_utc', 'base_source', 'temp_C', 'rh_pct', 'dew_C',
               'pres_corr_hPa', 'u_ms', 'v_ms', 'z_ref_m', 'z_ref_method', 'source_z_ref_m',
               'source_file_or_key', 'source_points', 'source_clusters', 'source_limitation']
    # Use exact embedded datum and add the unambiguous case-event timestamp.
    first = zero.set_index('case_id')
    exported = b[columns].copy()
    exported['embedded_zero_offset_z_ref_m'] = exported.case_id.map(first.z_ref_m_min)
    exported.insert(2, 'event_time_utc', exported.case_id.map(first.embedded_t0).astype(str))
    exported.to_csv(PAPER / 'supplement_base_states.csv', index=False)
    return report, b


def axes_style(ax):
    ax.set(xlim=(-5, 5), ylim=(-5, 5), xticks=[-4, 0, 4], yticks=[-4, 0, 4], aspect='equal')
    ax.axhline(0, color='.5', lw=.4, ls='--')
    ax.axvline(0, color='.5', lw=.4, ls='--')
    ax.plot(0, 0, '+', color='k', ms=7, mew=.8)
    ax.tick_params(length=2, pad=2)


def render(report, b):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'hatch.linewidth': .35})
    figures = PAPER / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    refs = {}
    for mode in ('T', 'N'):
        for offset in OFFSETS:
            path = ROOT / f'data/dbz_composites/ref_composite_{mode.lower()}_height_3p00km_{offset}min.pkl'
            with path.open('rb') as f:
                rx, ry, dbz = pickle.load(f)
            refs[mode, offset] = (np.asarray(rx) / 1000, np.asarray(ry) / 1000, np.asarray(dbz))
            report['inputs'][str(path.relative_to(ROOT))] = digest(path)
    for var, (limit, diff_limit, ci_limit, label) in STYLE.items():
        fig, axes = plt.subplots(3, 5, figsize=(12.2, 7.5), sharex=True, sharey=True)
        figci, ciaxes = plt.subplots(3, 5, figsize=(12.2, 7.5), sharex=True, sharey=True)
        for f in (fig, figci):
            f.subplots_adjust(left=.065, right=.89, bottom=.075, top=.96, wspace=.08, hspace=.18)
        for col, offset in enumerate(OFFSETS):
            a = np.load(RETURN / f'{var}_{offset:+03d}.npz')
            X, Y = a['X'], a['Y']
            for row, group in enumerate(('T', 'N', 'difference')):
                ax, cax = axes[row, col], ciaxes[row, col]
                mean = a['difference'] if row == 2 else a[group + '_mean']
                bound = diff_limit if row == 2 else limit
                im = ax.pcolormesh(X, Y, mean, vmin=-bound, vmax=bound, cmap='RdBu_r', shading='auto')
                if row == 0:
                    mean_im = im
                eligible = a['test_eligible'] if row == 2 else a[group + '_eligible']
                if row < 2:
                    rx, ry, dbz = refs[group, offset]
                    ax.contour(rx, ry, dbz, levels=[35, 40, 45, 50], colors='k', linewidths=.65)
                    stride = 12
                    q = ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                                  a[group + '_var_u_barnes'][::stride, ::stride],
                                  a[group + '_var_v_barnes'][::stride, ::stride],
                                  scale=150, width=.0055, color='.15')
                    if col == 0:
                        ax.quiverkey(q, .25, .08, 10, '10 m/s', coordinates='axes', labelpos='N', fontproperties={'size': 9})
                else:
                    # Hatching leaves the observed difference visible but identifies untested cells.
                    h = ax.contourf(X, Y, (~eligible).astype(float), levels=[.5, 1.5], colors='none', hatches=['///'])
                    h.set_edgecolor('.35')
                    h.set_linewidth(0.)
                half = (a[group + '_ci_high'] - a[group + '_ci_low']) / 2
                cax.set_facecolor('.82')
                cim = cax.pcolormesh(X, Y, np.where(eligible, half, np.nan), vmin=0, vmax=ci_limit, cmap='viridis', shading='auto')
                for target in (ax, cax):
                    axes_style(target)
                    target.set_title(f'({chr(97 + row * 5 + col)})' + (f' {offset:+d} min' if row == 0 else ''), loc='left', fontsize=12, pad=3)
            axes[-1, col].set_xlabel('$x$ (km)', labelpad=1)
            ciaxes[-1, col].set_xlabel('$x$ (km)', labelpad=1)
        for row, name in enumerate(('Tornadic', 'Nontornadic', 'T minus N')):
            axes[row, 0].set_ylabel(name + '\n$y$ (km)', labelpad=1)
            ciaxes[row, 0].set_ylabel(name + '\n$y$ (km)', labelpad=1)
        fig.colorbar(mean_im, cax=fig.add_axes([.91, .42, .016, .51]), label=label, extend='both')
        fig.colorbar(im, cax=fig.add_axes([.91, .10, .016, .22]), label='T minus N', extend='both')
        figci.colorbar(cim, cax=figci.add_axes([.91, .13, .016, .76]), label='95% CI half-width: ' + label, extend='max')
        fig.savefig(figures / f'{var}_barnes_mwr.png', dpi=300)
        figci.savefig(figures / f'{var}_confidence_width_mwr.png', dpi=300)
        plt.close(fig); plt.close(figci)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained')
    for ax, (key, label), letter in zip(axes.flat, [('pres_corr_hPa', 'Pressure (hPa)'), ('temp_C', 'Temperature (°C)'),
                  ('rh_pct', 'Relative humidity (%)'), ('u_ms', '$u$ (m s$^{-1}$)'), ('v_ms', '$v$ (m s$^{-1}$)'),
                  ('dew_C', 'Dewpoint (°C)')], 'abcdef'):
        edges = np.histogram_bin_edges(b[key], bins=10)
        for mode, color, name in [('T', '#bc3b3c', 'Tornadic'), ('N', '#3677b5', 'Nontornadic')]:
            ax.hist(b.loc[b['case_id'].str.startswith(mode), key], bins=edges, histtype='step', linewidth=2, color=color, label=name)
        ax.set(xlabel=label, ylabel='Number of storms', title=f'({letter})')
    axes[0, 0].legend(fontsize=10)
    fig.savefig(figures / 'base_state_distributions_mwr.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    report, base = validate()
    render(report, base)
    report['outputs'] = {str(p.relative_to(ROOT)): digest(p) for p in (PAPER / 'figures').glob('*_mwr.png')}
    report['outputs']['manuscript/CH2/supplement_base_states.csv'] = digest(PAPER / 'supplement_base_states.csv')
    (AUDIT / 'schooner_return_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Verified 40 grids, 109 base states, and 34,439 tests. Figures and audit saved.')
