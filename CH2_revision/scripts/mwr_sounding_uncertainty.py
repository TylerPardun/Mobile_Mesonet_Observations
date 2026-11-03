#!/usr/bin/env python3
"""Reproduce CH2 sounding means and quantify sampling uncertainty by storm day.

The estimand is the equal-case mean of the existing within-case mean metrics.
No new parcel calculations or case selections are made here. Resampling whole
1200–1200 UTC days preserves dependence among storms sampled on the same day,
including days represented in both outcome groups. Intervals are pointwise,
descriptive 95% percentile intervals, not simultaneous significance tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("SBCAPE", r"Surface-based CAPE (J kg$^{-1}$)"),
    ("SBCIN", r"Surface-based CIN (J kg$^{-1}$)"),
    ("MLCAPE", r"Mixed-layer CAPE (J kg$^{-1}$)"),
    ("MLCIN", r"Mixed-layer CIN (J kg$^{-1}$)"),
    ("MUCAPE", r"Most-unstable CAPE (J kg$^{-1}$)"),
    ("MUCIN", r"Most-unstable CIN (J kg$^{-1}$)"),
    ("LCL_z", "LCL height (m)"),
    ("LCL_t", r"LCL temperature ($^{\circ}$C)"),
    ("SHR05_mag", r"0--500~m shear (m s$^{-1}$)"),
    ("SHR01_mag", r"0--1~km shear (m s$^{-1}$)"),
    ("SHR03_mag", r"0--3~km shear (m s$^{-1}$)"),
    ("SHR06_mag", r"0--6~km shear (m s$^{-1}$)"),
    ("SRH05", r"0--500~m SRH (m$^2$ s$^{-2}$)"),
    ("SRH_05_bunkers", r"0--500~m SRH* (m$^2$ s$^{-2}$)"),
    ("SRH01", r"0--1~km SRH (m$^2$ s$^{-2}$)"),
    ("SRH_01_bunkers", r"0--1~km SRH* (m$^2$ s$^{-2}$)"),
    ("SRH03", r"0--3~km SRH (m$^2$ s$^{-2}$)"),
    ("SRH_03_bunkers", r"0--3~km SRH* (m$^2$ s$^{-2}$)"),
    ("EFF_depth_m", "Effective inflow depth (m)"),
    ("EBS_mag", r"Effective bulk shear (m s$^{-1}$)"),
    ("SRH_eff", r"Effective-layer SRH (m$^2$ s$^{-2}$)"),
    ("SRH_effective_bunkers", r"Effective-layer SRH* (m$^2$ s$^{-2}$)"),
    ("SM_spd", r"Storm motion speed (m s$^{-1}$)"),
    ("BNK_RM_spd", r"Storm motion speed* (m s$^{-1}$)"),
]


def case_metrics(root=ROOT):
    data = root / "data"
    tr = pd.read_excel(data / "transect_times.xlsx", sheet_name="transect_times")
    if tr.CaseID.duplicated().any():
        raise ValueError("Duplicate case IDs")
    tr["t0"] = pd.to_datetime(tr.Tstart, format="%Y%m%d-%H%M")
    tr["day"] = (tr.t0 - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    lookup = {}
    for row in tr.itertuples():
        for offset in range(-20, 21, 5):
            key = (row.t0 + pd.Timedelta(minutes=offset)).strftime("%Y%m%d%H%M%S")
            mk = (row.CaseID[0], key)
            if mk in lookup:
                raise ValueError(f"Ambiguous case mapping: {mk}")
            lookup[mk] = (row.CaseID, row.day)
    with (data / "sounding_metrics.pkl").open("rb") as f:
        metrics = pickle.load(f)
    with (data / "soundings_RAP_bunkers.pkl").open("rb") as f:
        bunkers = pickle.load(f)
    rows = []
    for mode in ("T", "N"):
        for key, values in metrics[mode].items():
            case, day = lookup[(mode, str(key))]
            merged = dict(values or {})
            merged.update(bunkers[mode].get(key) or {})
            row = dict(case_id=case, day=day, mode=mode, analysis_key=key)
            for name, _ in METRICS:
                v = merged.get(name, np.nan)
                if name == "SM_spd":
                    v = merged.get("SM_spd_ms", merged.get("SM_spd", np.nan) * 0.514444)
                elif name == "BNK_RM_spd":
                    v = v * 0.514444
                a = np.asanyarray(v)
                row[name] = np.nan if np.ma.is_masked(a) else float(a.item())
            rows.append(row)
    analyses = pd.DataFrame(rows)
    result = analyses.groupby(["case_id", "day", "mode"], as_index=False)[
        [m for m, _ in METRICS]
    ].mean()
    if len(result) != 109 or result.groupby("mode").size().to_dict() != {"N": 57, "T": 52}:
        raise ValueError("Unexpected case coverage")
    return result, analyses


def cluster_replicates(values, modes, day_codes, draws):
    """Ratio of weighted case sums/counts; days, never individual rows, resampled."""
    n_days = draws.shape[1]
    reps = []
    for mode in ("T", "N"):
        keep = (modes == mode) & np.isfinite(values)
        sums = np.bincount(day_codes[keep], weights=values[keep], minlength=n_days)
        counts = np.bincount(day_codes[keep], minlength=n_days)
        denom = draws @ counts
        if np.any(denom == 0):
            raise ValueError("A bootstrap sample contains no finite cases for one group")
        reps.append((draws @ sums) / denom)
    return np.column_stack([reps[0], reps[1], reps[0] - reps[1]])


def run(output, n_resamples=50000, seed=20261103):
    output.mkdir(parents=True, exist_ok=True)
    cases, analyses = case_metrics()
    cases.to_csv(output / "sounding_case_metrics.csv", index=False)
    day_codes, days = pd.factorize(cases.day, sort=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_resamples)
    previous = pd.read_csv(ROOT / "scripts/sounding_metrics_table.csv").set_index("variable")
    rows, latex, convergence = [], [], []
    for metric, label in METRICS:
        v = cases[metric].to_numpy(float)
        means = [v[cases["mode"].to_numpy() == m] for m in ("T", "N")]
        means = [np.nanmean(a) for a in means]
        # This is an independent reproduction check, not a rounded-table check.
        np.testing.assert_allclose(means, previous.loc[metric, ["T_mean", "N_mean"]].to_numpy(float), atol=1e-9, rtol=0)
        reps = cluster_replicates(v, cases["mode"].to_numpy(), day_codes, draws)
        ci = np.quantile(reps, [0.025, 0.975], axis=0)
        half_ci = np.quantile(reps[:n_resamples//2], [0.025, 0.975], axis=0)
        row = dict(variable=metric, T_mean=means[0], N_mean=means[1], difference=means[0]-means[1])
        for j, group in enumerate(("T", "N", "difference")):
            row[group + "_ci_low"] = ci[0, j]
            row[group + "_ci_high"] = ci[1, j]
            convergence.append(dict(variable=metric, group=group, max_endpoint_change=float(abs(ci[:,j]-half_ci[:,j]).max()), interval_width=float(ci[1,j]-ci[0,j])))
        rows.append(row)
        estimates = means + [means[0]-means[1]]
        cells = [f"{x:.1f} [{ci[0,j]:.1f}, {ci[1,j]:.1f}]" for j,x in enumerate(estimates)]
        latex.append(label + " & " + " & ".join(cells) + r" \\")
    pd.DataFrame(rows).to_csv(output / "sounding_uncertainty.csv", index=False)
    pd.DataFrame(convergence).to_csv(output / "sounding_bootstrap_convergence.csv", index=False)
    (output / "sounding_table_rows.tex").write_text("\n".join(latex) + "\n")
    inputs = [ROOT/"data"/n for n in ("sounding_metrics.pkl", "soundings_RAP_bunkers.pkl", "transect_times.xlsx")]
    inputs += [Path(__file__), ROOT/"scripts/sounding_metrics_table.csv"]
    metadata = dict(seed=seed, generator="numpy.default_rng/PCG64", resamples=n_resamples,
        cluster="1200–1200 UTC day of case t0", n_days=len(days), n_cases=len(cases),
        analysis_rows=analyses.groupby("mode").size().to_dict(),
        case_counts=cases.groupby("mode").size().to_dict(),
        day_counts=cases.groupby("mode").day.nunique().to_dict(),
        interval="pointwise 95% percentile cluster bootstrap; not simultaneous tests",
        estimand="equal-case mean of within-case mean sounding metrics",
        validation="means reproduce the prior full-precision table within 1e-9",
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (output / "sounding_uncertainty_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ("n_cases","n_days","day_counts","resamples")},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"manuscript/CH2/revisions/mwr_revision_audit_20261103")
    parser.add_argument("--resamples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    run(args.output, args.resamples, args.seed)
