"""
supercell_mode_probs.py
-----------------------
Fuzzy-logic probability classifier for supercell mode (HP / CS / LP) based
on four inflow proximity sounding parameters.

Membership thresholds and weights are derived from:

  LCL height (weight 0.45):
    Rasmussen & Blanchard (1998, WAF 13, 1148–1164) — Table 1 LCL distributions
    Thompson et al. (2003, WAF 18, 1243–1261)       — Fig. 4 LCL by storm type

  Precipitable water (weight 0.30):
    Brooks et al. (1994, MWR 122, 2217–2236)        — moisture in tornado envs
    Doswell & Evans (2003, Atmos. Res. 67–68, 199)  — moisture and storm morphology
    (specific PW thresholds reflect operational guidance, not a single paper)

  Surface dewpoint (weight 0.25):
    Bluestein & Parks (1983, MWR 111, 2034–2046)    — LP supercell environments
    Moller et al. (1994, WAF 9, 327–347)             — original HP/LP/CS definitions

  MLCAPE: used only as a confidence gate; not a mode discriminator per the literature
    (Rasmussen & Blanchard 1998 — CAPE does not cleanly separate HP/LP/CS)

Usage
-----
    from supercell_mode_probs import classify_mode
    result = classify_mode(lclhght, PW, sfc_td, mlcape)
    print(result)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Trapezoidal fuzzy membership function
# ---------------------------------------------------------------------------
def _trap(x, a, b, c, d):
    """
    Trapezoidal membership: 0 outside [a,d], rises linearly a→b,
    full (1) on [b,c], falls linearly c→d.
    Handles x < a and x > d as 0, clamps within [0,1].
    """
    if x <= a:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif x <= c:
        return 1.0
    elif x <= d:
        return (d - x) / (d - c) if d > c else 0.0
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Per-variable membership tables
# (a, b, c, d) = trapezoidal corners; INF = 9999 (open-ended)
# ---------------------------------------------------------------------------
INF = 9999.0

#  LCL height (m AGL)
#    HP  : full for LCL < 700 m, tapers to 0 by 1200 m
#    CS  : rises from 600 m, peaks 900–1400 m, tapers to 0 by 1900 m
#    LP  : rises from 1200 m, full above 1700 m
_LCL = {
    'HP': (0,     0,    700,  1200),
    'CS': (600,   900,  1400, 1900),
    'LP': (1200,  1700, INF,  INF ),
}

#  Precipitable water (inches)
#    HP  : rises from 0.9 in, full above 1.25 in
#    CS  : rises from 0.6 in, peaks 0.80–1.20 in, tapers to 0 by 1.5 in
#    LP  : full below 0.60 in, tapers to 0 by 0.9 in
_PW = {
    'HP': (0.90,  1.25, INF,  INF ),
    'CS': (0.60,  0.80, 1.20, 1.50),
    'LP': (0,     0,    0.60, 0.90),
}

#  Surface dewpoint (°F)
#    HP  : rises from 57 °F, full above 63 °F
#    CS  : rises from 50 °F, peaks 55–63 °F, tapers to 0 by 68 °F
#    LP  : full below 50 °F, tapers to 0 by 57 °F
_TD = {
    'HP': (57.0,  63.0, INF,  INF ),
    'CS': (50.0,  55.0, 63.0, 68.0),
    'LP': (0,     0,    50.0, 57.0),
}

# Feature weights (must sum to 1.0)
_WEIGHTS = {'lcl': 0.45, 'pw': 0.30, 'td': 0.25}


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------
def classify_mode(lclhght_m, pw_in, sfc_td_f, mlcape_jkg):
    """
    Compute fuzzy-logic supercell mode probabilities.

    Parameters
    ----------
    lclhght_m  : float  LCL height (m AGL)  — pcl.lclhght from parcelx(flag=1)
    pw_in      : float  Precipitable water (inches) — params.precip_water(prof)
    sfc_td_f   : float  Surface dewpoint (°F) — (dwpc[0] * 9/5) + 32
    mlcape_jkg : float  Mixed-layer CAPE (J/kg) — ml_pcl.bplus from parcelx(flag=4)
                        Used only as a confidence gate; does not shift mode probs.

    Returns
    -------
    dict with keys:
        'HP'         : float  probability [0–1]
        'CS'         : float  probability [0–1]
        'LP'         : float  probability [0–1]
        'most_likely': str    highest-probability mode label
        'confidence' : float  [0–1], penalised for low MLCAPE or degenerate input
        'flag'       : str    warning message (empty if none)
    """
    flag = ''

    # Guard against NaN inputs
    inputs = [lclhght_m, pw_in, sfc_td_f, mlcape_jkg]
    if any(not np.isfinite(v) for v in inputs):
        return {'HP': np.nan, 'CS': np.nan, 'LP': np.nan,
                'most_likely': 'unknown', 'confidence': 0.0,
                'flag': 'One or more inputs are NaN/inf'}

    # MLCAPE confidence gate (not a mode discriminator — Rasmussen & Blanchard 1998)
    if mlcape_jkg < 500:
        flag = f'MLCAPE = {mlcape_jkg:.0f} J/kg: marginal instability for supercell'
        cape_conf = max(0.0, mlcape_jkg / 500.0)
    else:
        cape_conf = min(1.0, mlcape_jkg / 1500.0)   # full confidence at 1500+ J/kg

    # Compute membership scores for each mode across the three variables
    scores = {}
    for mode in ('HP', 'CS', 'LP'):
        m_lcl = _trap(lclhght_m, *_LCL[mode])
        m_pw  = _trap(pw_in,     *_PW[mode])
        m_td  = _trap(sfc_td_f,  *_TD[mode])

        scores[mode] = (_WEIGHTS['lcl'] * m_lcl +
                        _WEIGHTS['pw']  * m_pw  +
                        _WEIGHTS['td']  * m_td)

    total = sum(scores.values())

    # Degenerate case: all memberships are effectively zero
    if total < 1e-6:
        flag = (flag + ' | All membership scores near zero — '
                'environment does not match HP/LP/CS profiles').lstrip(' | ')
        return {'HP': 1/3, 'CS': 1/3, 'LP': 1/3,
                'most_likely': 'unknown', 'confidence': 0.0, 'flag': flag}

    probs  = {m: scores[m] / total for m in ('HP', 'CS', 'LP')}
    winner = max(probs, key=probs.get)

    # Overall confidence: how decisive is the winner, scaled by CAPE gate
    decisiveness = probs[winner] - sorted(probs.values())[-2]   # margin over 2nd place
    confidence   = round(min(1.0, decisiveness * 2.0) * cape_conf, 3)

    return {
        'HP':          round(probs['HP'], 3),
        'CS':          round(probs['CS'], 3),
        'LP':          round(probs['LP'], 3),
        'most_likely': winner,
        'confidence':  confidence,
        'flag':        flag,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: run over all cases in tstats
# ---------------------------------------------------------------------------
def classify_all_cases(tstats, mesonet_data, mode_key='T'):
    """
    Loop over every case in tstats['case_mean_profiles'] and return a dict
    keyed by caseID integer.

    Parameters
    ----------
    tstats       : dict  output of compute_sounding_stats()
    mesonet_data : dict  the main mesonet pickle (for caseID lookup)
    mode_key     : str   'T' or 'N'

    Returns
    -------
    dict keyed by int caseID -> classify_mode result dict
    """
    import sharppy.sharptab.params as params

    results = {}
    for cid_str, prof in tstats['case_mean_profiles'].items():
        try:
            # Surface parcel LCL
            pcl      = params.parcelx(prof, flag=1)
            lclhght  = float(pcl.lclhght)

            # Precipitable water (inches)
            pw       = float(params.precip_water(prof))

            # Surface dewpoint (°F)
            sfc_td   = float((np.array(prof.dwpc)[0] * (9/5)) + 32)

            # Mixed-layer CAPE
            ml_pcl   = params.parcelx(prof, flag=4)
            mlcape   = float(ml_pcl.bplus)

            results[int(cid_str)] = classify_mode(lclhght, pw, sfc_td, mlcape)
            results[int(cid_str)]['lclhght_m'] = round(lclhght, 1)
            results[int(cid_str)]['pw_in']     = round(pw, 3)
            results[int(cid_str)]['sfc_td_f']  = round(sfc_td, 1)
            results[int(cid_str)]['mlcape']    = round(mlcape, 1)

        except Exception as e:
            results[int(cid_str)] = {'flag': f'Error: {e}', 'most_likely': 'unknown'}

    return results


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_result(cid, res):
    bar_chars = 30
    print(f'\n  Case {cid}  →  most likely: {res["most_likely"]}  '
          f'(confidence {res["confidence"]:.2f})')
    for mode in ('HP', 'CS', 'LP'):
        p = res.get(mode, np.nan)
        if np.isfinite(p):
            bar = '█' * int(round(p * bar_chars))
            print(f'    {mode}  {p:.3f}  {bar}')
    if res.get('flag'):
        print(f'    ⚠  {res["flag"]}')
    if 'lclhght_m' in res:
        print(f'    inputs: LCL={res["lclhght_m"]:.0f} m  '
              f'PW={res["pw_in"]:.2f} in  '
              f'Td={res["sfc_td_f"]:.1f}°F  '
              f'MLCAPE={res["mlcape"]:.0f} J/kg')


# ---------------------------------------------------------------------------
# Example — paste your existing variable block and call classify_mode()
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pickle
    from datetime import datetime
    import sharppy.sharptab.params as params

    # --- Load your data (adjust paths as needed) ---
    with open('monte_carlo_stuff.pkl', 'rb') as f:
        bundle = pickle.load(f)
    mesonet_data = bundle[0]

    # tstats would come from compute_sounding_stats() — substitute your object here
    # For demonstration, run over the soundings.pkl profiles instead
    with open('../data/soundings_RAP.pkl', 'rb') as f:
        snd = pickle.load(f)

    print('=== Supercell mode classification demo ===')
    print('  (using soundings.pkl profiles as stand-in for tstats case_mean_profiles)\n')

    seen = set()
    for i, (prof, camp) in enumerate(zip(snd['profs'], snd['campagin'])):
        if camp in seen or i > 40:
            continue
        seen.add(camp)
        try:
            pcl     = params.parcelx(prof, flag=1)
            lclhght = float(pcl.lclhght)
            pw      = float(params.precip_water(prof))
            sfc_td  = float((np.array(prof.dwpc)[0] * 9/5) + 32)
            ml_pcl  = params.parcelx(prof, flag=4)
            mlcape  = float(ml_pcl.bplus)

            res = classify_mode(lclhght, pw, sfc_td, mlcape)
            res.update({'lclhght_m': round(lclhght, 1), 'pw_in': round(pw, 3),
                        'sfc_td_f': round(sfc_td, 1), 'mlcape': round(mlcape, 1)})
            print_result(f'{camp} (sounding {i})', res)
        except Exception as e:
            print(f'  Sounding {i}: error — {e}')


'''
from sharppy.sharptab import interp as sharpinterp
from supercell_mode_probs import classify_mode, print_result

mode = 'N'
caseID_val = 47
for caseID_val in range(48):
    try:
    
        caseid_time = datetime.strptime(mesonet_data[mode][0]['t0'][np.where(mesonet_data[mode][0]['caseID']==caseID_val)[0]][0], '%Y%m%d%H%M')


        if mode == 'T':
            prof = tstats['case_mean_profiles'][str(caseID_val)]
        else:
            prof = nstats['case_mean_profiles'][str(caseID_val)]

        #LCL height
        pcl = params.parcelx(prof,flag=1)
        lclhght = pcl.lclhght

        #Precipitable water
        PW = params.precip_water(prof)

        #Surface dewpoint
        sfc_td = (np.array(prof.dwpc)[0]*(9/5)) + 32

        #MLCAPE
        ml_pcl = params.parcelx(prof, flag=4)
        mlcape = ml_pcl.bplus

        result = classify_mode(lclhght, PW, sfc_td, mlcape)
        print()
        print(caseid_time)
        print_result(caseID_val, result)
    except:
        pass

'''