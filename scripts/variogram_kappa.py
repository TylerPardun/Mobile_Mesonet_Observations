#!/usr/bin/env python3
"""
variogram_kappa.py
------------------
Derives the Barnes smoothing parameter kappa from the spatial
semivariogram of the mobile mesonet composite observations.

Theoretical basis
-----------------
The Barnes weight function w(r) = exp(-r^2 / kappa) is a Gaussian
kernel whose e-folding distance is Re = sqrt(kappa).  The Gaussian
semivariogram model has the same functional form:

    gamma(h) = C0 + C1 * (1 - exp(-h^2 / a^2))

where 'a' is the spatial correlation length (e-folding distance).
Direct correspondence: kappa = a^2.

Key design choices
------------------
1. WITHIN-CASE pairs only.  Cross-case pairs (obs from different
   storms) are statistically independent and contribute only to the
   sill at all lags, inflating the nugget and biasing the range.
   The caseID field is used to restrict pairs to the same event.

2. On-the-fly binning.  Bin sums are accumulated during the loop
   (no large array concatenation), keeping memory and runtime low.

3. pairs computed once per case per time, reused across all 7 vars.

Runtime: ~15-25 s on a laptop (no external parallelism needed).
"""

import matplotlib
matplotlib.use('Agg')          # non-interactive; prevents GUI hang

import pickle
import warnings
import time

import numpy as np
from scipy.spatial  import cKDTree
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as ticker

# ── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_PATH    = 'monte_carlo_stuff.pkl'
DOMAIN_KM    = 10.5      # clip obs to ±DOMAIN_KM (km) before analysis
MAX_LAG      = 7.0      # km — max variogram lag
N_LAGS       = 20       # number of equal-width lag bins
MIN_PAIRS    = 30       # minimum pairs per bin to include in fit
N_BOOT       = 400      # bootstrap resamples for 95% CI on kappa
GAMMA_BARNES = 0.3      # two-pass gamma (for response diagnostics only)
SEED         = 42

VARIABLES = {
    'the':       r"$\theta'_e$ (K)",
    'thv':    r"$\theta'_v$ (K)",
    'thw':       r"$\theta'_w$ (K)",
    'qv':        r"$q'_v$ (kg kg$^{-1}$)",
    'pres_corr': r"$p'$ (hPa)",
    'sru':       r"$u'_{SR}$ (m s$^{-1}$)",
    'srv':       r"$v'_{SR}$ (m s$^{-1}$)",
}
# ───────────────────────────────────────────────────────────────────────────


# ── VARIOGRAM MODEL ─────────────────────────────────────────────────────────
def gaussian_vgm(h, C0, C1, a):
    """gamma(h) = C0 + C1*(1 - exp(-h^2/a^2))  →  kappa = a^2"""
    return C0 + C1 * (1.0 - np.exp(-(h / a) ** 2))


# ── FIT GAUSSIAN VARIOGRAM ──────────────────────────────────────────────────
def fit_gaussian(h_bin, g_bin, n_bin):
    """
    Weighted least-squares Gaussian variogram fit.
    Only bins with n >= MIN_PAIRS are used.
    Returns (C0, C1, a) or None on failure.
    """
    mask = n_bin >= MIN_PAIRS
    if mask.sum() < 4:
        return None

    h = h_bin[mask];  g = g_bin[mask];  w = np.sqrt(n_bin[mask])

    C_top = float(np.nanmax(g))
    C0_0  = max(float(np.nanmin(g)), 0.0)
    C1_0  = max(C_top - C0_0, 1e-8)
    a_0   = MAX_LAG / 4.0

    bounds = ([0,      0,      0.05  ],
              [C_top,  3*C_top, MAX_LAG])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            popt, _ = curve_fit(gaussian_vgm, h, g,
                                p0=[C0_0, C1_0, a_0],
                                sigma=1.0 / w,
                                bounds=bounds, maxfev=8000)
            return popt
        except Exception:
            return None


# ── BOOTSTRAP CI ────────────────────────────────────────────────────────────
def bootstrap_kappa_ci(h_bin, g_bin, n_bin, n_boot, seed):
    """
    Resample the binned variogram N_BOOT times and refit to get
    a 95% CI on kappa = a^2.  Returns (kappa_lo, kappa_hi).
    """
    mask = n_bin >= MIN_PAIRS
    if mask.sum() < 4:
        return np.nan, np.nan

    h_ok = h_bin[mask];  g_ok = g_bin[mask];  n_ok = n_bin[mask]
    nb   = int(mask.sum())
    rng  = np.random.default_rng(seed)
    kaps = []

    for _ in range(n_boot):
        idx  = rng.choice(nb, nb, replace=True)
        popt = fit_gaussian(h_ok[idx], g_ok[idx], n_ok[idx])
        if popt is not None:
            kaps.append(popt[2] ** 2)

    kaps = np.array(kaps)
    if kaps.size < 20:
        return np.nan, np.nan
    return float(np.percentile(kaps, 2.5)), float(np.percentile(kaps, 97.5))


# ── BARNES TWO-PASS RESPONSE ────────────────────────────────────────────────
def two_pass_response(L, kappa, gamma=GAMMA_BARNES):
    D = np.exp(-(np.pi ** 2) * kappa / L ** 2)
    return D + D ** gamma - D ** (1.0 + gamma)


def L_halfpower(kappa):
    try:
        return brentq(lambda L: two_pass_response(L, kappa) - 0.5, 1e-5, 500.0)
    except Exception:
        return np.nan


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    t_wall = time.perf_counter()

    # ── Load ───────────────────────────────────────────────────────────────
    print("Loading data …")
    with open(DATA_PATH, 'rb') as f:
        bundle = pickle.load(f)

    mesonet_data = bundle[0]
    modes        = list(mesonet_data.keys())
    times        = list(mesonet_data[modes[0]].keys())
    var_keys     = list(VARIABLES.keys())
    nv           = len(var_keys)

    print(f"  Modes : {modes}  |  Times : {[int(t) for t in times]}")
    print(f"  Vars  : {var_keys}\n")

    # ── Pre-allocate on-the-fly bin accumulators ────────────────────────────
    # gsum[dest][vk]  = array of shape (N_LAGS,) — running sum of gamma
    # ncount[dest][vk]= array of shape (N_LAGS,) — running pair count
    dests = modes + ['ALL']
    gsum   = {d: {vk: np.zeros(N_LAGS) for vk in var_keys} for d in dests}
    ncount = {d: {vk: np.zeros(N_LAGS, dtype=np.int64)
                  for vk in var_keys} for d in dests}

    lag_edges   = np.linspace(0.0, MAX_LAG, N_LAGS + 1)
    lag_centres = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    # ── Main loop: within-case pairs, on-the-fly binning ───────────────────
    print("Computing within-case variogram pairs …")
    n_case_slices  = 0
    n_total_pairs  = 0

    for mode in modes:
        for cnx in times:
            rec      = mesonet_data[mode][cnx]
            x_all    = np.asarray(rec['rdx'],    float)
            y_all    = np.asarray(rec['rdy'],    float)
            cid_all  = np.asarray(rec['caseID'], int)

            # clip to analysis domain
            inside = ((np.abs(x_all) <= DOMAIN_KM) &
                      (np.abs(y_all) <= DOMAIN_KM) &
                      np.isfinite(x_all) & np.isfinite(y_all))

            if inside.sum() < 4:
                continue

            x_d   = x_all[inside]
            y_d   = y_all[inside]
            cid_d = cid_all[inside]

            # pre-extract variables for this mode/time (avoid repeated indexing)
            var_arrays = {vk: np.asarray(rec[vk], float)[inside]
                          for vk in var_keys}

            # iterate over individual cases
            unique_cases = np.unique(cid_d)
            for cid in unique_cases:
                c_mask = cid_d == cid
                if c_mask.sum() < 4:
                    continue

                xc = x_d[c_mask];  yc = y_d[c_mask]

                # find within-case pairs
                tree  = cKDTree(np.column_stack([xc, yc]))
                pairs = tree.query_pairs(MAX_LAG, output_type='ndarray')

                if len(pairs) == 0:
                    continue

                ii, jj  = pairs[:, 0], pairs[:, 1]
                h_pairs = np.hypot(xc[ii] - xc[jj], yc[ii] - yc[jj])

                # bin index for each pair (vectorised)
                bin_idx = np.clip(
                    np.searchsorted(lag_edges[1:], h_pairs), 0, N_LAGS - 1
                )
                bc = np.bincount(bin_idx, minlength=N_LAGS)   # pair counts

                # accumulate gamma for each variable (pairs reused)
                for vk in var_keys:
                    zc  = var_arrays[vk][c_mask]
                    ok  = np.isfinite(zc[ii]) & np.isfinite(zc[jj])
                    if ok.sum() == 0:
                        continue

                    g_ok  = 0.5 * (zc[ii[ok]] - zc[jj[ok]]) ** 2
                    bi_ok = bin_idx[ok]

                    gs = np.bincount(bi_ok, weights=g_ok, minlength=N_LAGS)
                    nc = np.bincount(bi_ok, minlength=N_LAGS)

                    for dest in [mode, 'ALL']:
                        gsum[dest][vk]   += gs
                        ncount[dest][vk] += nc.astype(np.int64)

                n_case_slices += 1
                n_total_pairs += len(pairs)

    print(f"  {n_case_slices} case-time slices processed, "
          f"{n_total_pairs:,} total within-case pairs\n")

    # ── Compute empirical variogram curves ─────────────────────────────────
    # gamma_mean[dest][vk] = gsum / ncount (where ncount > 0)
    emp = {}
    for dest in dests:
        emp[dest] = {}
        for vk in var_keys:
            ns = ncount[dest][vk].astype(float)
            gs = gsum[dest][vk]
            valid = ns > 0
            h_out = lag_centres[valid]
            g_out = gs[valid] / ns[valid]
            n_out = ns[valid]
            emp[dest][vk] = (h_out, g_out, n_out)

    # ── Fit and report ─────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'Variable':<14} {'Mode':<5} {'a (km)':<9} "
          f"{'kappa (km^2)':<14} {'L50_2p (km)':<13} {'95% CI kappa'}")
    print("-" * 72)

    results = {}
    for vk in var_keys:
        results[vk] = {}
        for dest in dests:
            h_bin, g_bin, n_bin = emp[dest][vk]
            popt = fit_gaussian(h_bin, g_bin, n_bin)
            if popt is None:
                continue

            C0, C1, a = popt
            kappa = a ** 2
            L50   = L_halfpower(kappa)

            if dest == 'ALL':
                k_lo, k_hi = bootstrap_kappa_ci(h_bin, g_bin, n_bin,
                                                N_BOOT, SEED)
                ci_str = (f"[{k_lo:.3f}, {k_hi:.3f}]"
                          if np.isfinite(k_lo) else "—")
            else:
                k_lo, k_hi = np.nan, np.nan
                ci_str = "—"

            results[vk][dest] = dict(
                popt=popt, kappa=kappa, L50=L50,
                k_lo=k_lo, k_hi=k_hi,
                h_bin=h_bin, g_bin=g_bin, n_bin=n_bin,
            )
            print(f"  {vk:<12} {dest:<5} {a:<9.3f} {kappa:<14.4f} "
                  f"{L50:<13.3f} {ci_str}")

    print("=" * 72)

    # ── Consensus kappa ────────────────────────────────────────────────────
    kappas_all = [results[vk]['ALL']['kappa']
                  for vk in var_keys if 'ALL' in results[vk]]
    kappa_med  = float(np.median(kappas_all))
    kappa_q25  = float(np.percentile(kappas_all, 25))
    kappa_q75  = float(np.percentile(kappas_all, 75))
    Re_med     = np.sqrt(kappa_med)
    L50_med    = L_halfpower(kappa_med)

    print(f"\nConsensus κ  (median across all variables, pooled modes):")
    print(f"  κ          = {kappa_med:.4f} km²"
          f"   [IQR  {kappa_q25:.4f} – {kappa_q75:.4f} km²]")
    print(f"  Re = √κ    = {Re_med:.4f} km   (e-folding radius)")
    print(f"  rmax = 3Re = {3*Re_med:.4f} km   (search radius)")
    print(f"  Two-pass (γ={GAMMA_BARNES}) 50% at L = {L50_med:.4f} km")

    t_end = time.perf_counter()
    print(f"\nTotal runtime: {t_end - t_wall:.1f} s")

    # ── Figure ─────────────────────────────────────────────────────────────
    ncol = 4
    nrow = int(np.ceil(nv / ncol)) + 1

    fig = plt.figure(figsize=(5.0 * ncol, 4.2 * nrow))
    gs  = gridspec.GridSpec(nrow, ncol, figure=fig,
                            hspace=0.52, wspace=0.38)

    h_smooth = np.linspace(0, MAX_LAG, 300)
    clr  = {'T': '#d62728', 'N': '#1f77b4', 'ALL': 'k'}
    lbl  = {'T': 'Tornadic', 'N': 'Nontornadic', 'ALL': 'Pooled'}

    for iv, vk in enumerate(var_keys):
        ax = fig.add_subplot(gs[iv // ncol, iv % ncol])

        for dest in ['T', 'N', 'ALL']:
            if dest not in results[vk]:
                continue
            res       = results[vk][dest]
            C0, C1, a = res['popt']
            lw        = 2.5 if dest == 'ALL' else 1.5
            alp       = 1.0 if dest == 'ALL' else 0.60
            ls        = '-'  if dest == 'ALL' else '--'
            sz        = res['n_bin'] / res['n_bin'].max() * 55 + 15

            ax.scatter(res['h_bin'], res['g_bin'], s=sz,
                       color=clr[dest], alpha=alp, zorder=4,
                       label=f"{lbl[dest]}  κ={res['kappa']:.3f} km²")
            ax.plot(h_smooth, gaussian_vgm(h_smooth, C0, C1, a),
                    color=clr[dest], lw=lw, ls=ls, alpha=alp, zorder=3)

            if dest == 'ALL':
                ax.axvline(a, color='k', lw=1.2, ls=':',
                           label=f'a = {a:.3f} km')
                if np.isfinite(res['k_lo']):
                    ax.axvspan(np.sqrt(res['k_lo']), np.sqrt(res['k_hi']),
                               color='k', alpha=0.10)

        ax.set_xlabel('Lag h (km)', fontsize=10)
        ax.set_ylabel(r'$\hat{\gamma}(h)$', fontsize=10)
        ax.set_title(VARIABLES[vk], fontsize=11)
        ax.set_xlim(0, MAX_LAG);  ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc='upper left')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ── Summary bar panel ─────────────────────────────────────────────────
    ax_s  = fig.add_subplot(gs[nrow - 1, :])
    y_pos = np.arange(nv)
    kv, klo_err, khi_err, lv = [], [], [], []

    for vk in var_keys:
        lv.append(VARIABLES[vk])
        if 'ALL' in results[vk]:
            r = results[vk]['ALL']
            kv.append(r['kappa'])
            klo_err.append(r['kappa'] - r['k_lo'] if np.isfinite(r['k_lo']) else 0)
            khi_err.append(r['k_hi'] - r['kappa'] if np.isfinite(r['k_hi']) else 0)
        else:
            kv.append(np.nan);  klo_err.append(0);  khi_err.append(0)

    kv = np.array(kv)
    ax_s.barh(y_pos, kv, xerr=np.array([klo_err, khi_err]),
              color='steelblue', alpha=0.72, capsize=5, height=0.55,
              label='Per-variable κ  (error bars = 95% CI)')
    ax_s.axvline(kappa_med, color='k', lw=2.0, ls='--',
                 label=f'Consensus κ = {kappa_med:.3f} km²')
    ax_s.axvspan(kappa_q25, kappa_q75, color='k', alpha=0.10,
                 label=f'IQR [{kappa_q25:.3f}, {kappa_q75:.3f}] km²')
    ax_s.set_yticks(y_pos);  ax_s.set_yticklabels(lv, fontsize=10)
    ax_s.set_xlabel(r'$\kappa = a^2$ (km²)', fontsize=11)
    ax_s.set_title(
        r'Derived Barnes $\kappa$ per variable'
        r'  —  Gaussian variogram, within-case pairs only, pooled modes',
        fontsize=11)
    ax_s.legend(fontsize=10, loc='lower right')
    ax_s.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.savefig('variogram_kappa_results.png', dpi=200, bbox_inches='tight')
    print("Figure saved → variogram_kappa_results.png")


if __name__ == '__main__':
    main()
