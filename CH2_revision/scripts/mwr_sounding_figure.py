#!/usr/bin/env python3
"""Plot the saved CH2 case-mean profiles with storm-day bootstrap intervals.

Input winds are already rotated/storm-relative, as in the notebook export.
Only their stored SHARPpy knot units are converted to m/s. No new rotation,
storm-motion subtraction, sounding selection, or parcel calculation is made.
"""
from pathlib import Path
import hashlib
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from metpy.plots import SkewT
from metpy.units import units

from mwr_sounding_uncertainty import ROOT, case_metrics


class SavedProfile:
    """Read array attributes of an existing SHARPpy profile without recomputing it."""
    pass


class ProfileReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sharppy.sharptab.profile" and name == "BasicProfile":
            return SavedProfile
        return super().find_class(module, name)


def main():
    source = ROOT / "data/soundings_for_metrics.pkl"
    with source.open("rb") as f:
        profiles = ProfileReader(f).load()
    cases, _ = case_metrics()
    days = sorted(cases.day.unique())
    rng = np.random.default_rng(20261103)
    draws = rng.multinomial(len(days), np.full(len(days), 1/len(days)), size=50000)
    heights = np.arange(0., 15501., 10.)
    band_idx = np.arange(0, len(heights), 10)
    hodo_levels = [0, 100, 500, 1000, 2000, 3000, 6000, 8000]
    estimates, records, stored = {}, [], {}
    for mode in ("T", "N"):
        subset = cases[cases["mode"] == mode]
        if set(subset.case_id) != set(profiles[mode]):
            raise ValueError("Profile and metric cases do not match")
        values = {v: [] for v in ("pres", "tmpc", "dwpc", "u", "v")}
        for case in subset.case_id:
            prof = profiles[mode][case]["profs"]
            for var in values:
                arr = np.asarray(np.ma.filled(getattr(prof, var), np.nan), float)
                z = np.asarray(np.ma.filled(prof.hght, np.nan), float)
                valid = np.isfinite(arr) & np.isfinite(z)
                if np.any(np.diff(z[valid]) <= 0):
                    raise ValueError(f"Nonmonotonic heights: {case}")
                interpolated = np.interp(heights, z[valid], arr[valid], left=np.nan, right=np.nan)
                if var in ("u", "v"):
                    interpolated *= 0.514444  # identical conversion to original notebook
                values[var].append(interpolated)
        values = {v:np.asarray(a) for v,a in values.items()}
        codes = np.asarray([days.index(d) for d in subset.day])
        est = {v:np.nanmean(a, axis=0) for v,a in values.items()}
        for var in ("tmpc", "dwpc", "u", "v"):
            a = values[var][:, band_idx]
            sums = np.zeros((len(days), a.shape[1]))
            counts = np.zeros_like(sums)
            np.add.at(sums, codes, np.nan_to_num(a))
            np.add.at(counts, codes, np.isfinite(a))
            reps = (draws @ sums) / (draws @ counts)
            if not np.isfinite(reps).all():
                raise ValueError("Missing profile support in bootstrap samples")
            est[var+"_ci"] = np.quantile(reps, [.025, .975], axis=0)
            stored[mode+"_"+var+"_case"] = values[var]
            stored[mode+"_"+var+"_ci"] = est[var+"_ci"]
            for j,idx in enumerate(band_idx):
                records.append(dict(mode=mode, variable=var, height_m=heights[idx],
                    mean=est[var][idx], low=est[var+"_ci"][0,j], high=est[var+"_ci"][1,j]))
        estimates[mode] = est
    plt.rcParams.update({"font.size":13, "font.family":"DejaVu Sans"})
    fig = plt.figure(figsize=(13.5, 7.2))
    skew = SkewT(fig, rotation=35, rect=(.07,.12,.41,.81))
    skew.ax.set_ylim(1000,100)
    skew.ax.set_xlim(-35,45)
    skew.plot_dry_adiabats(alpha=.16, linewidth=.7)
    skew.plot_moist_adiabats(alpha=.14, linewidth=.7)
    skew.ax.set_title("(a) Composite soundings", loc="left", pad=12)
    ax = fig.add_axes((.57,.20,.40,.64))
    ax.set_title("(b) Rotated storm-relative hodographs",loc="left",pad=12)
    for mode, line, colors in (("T","-",("#c62828","#23763a")),("N","--",("#dd8a00","#2459bd"))):
        est = estimates[mode]
        name = "Tornadic" if mode=="T" else "Nontornadic"
        for var,color in zip(("tmpc","dwpc"),colors):
            skew.plot(est["pres"]*units.hPa, est[var]*units.degC,
                      color=color,linestyle=line,linewidth=1.8,
                      label=name+(' T' if var=='tmpc' else ' Td'))
            ci = est[var+"_ci"]
            skew.ax.fill_betweenx(est["pres"][band_idx],ci[0],ci[1],color=color,alpha=.15,linewidth=0)
        color = "#c62828" if mode=="T" else "#2459bd"
        use = heights<=8000
        ax.plot(est["u"][use],est["v"][use],line,color=color,lw=2,label=name)
        for h in hodo_levels:
            i=int(h/10);j=int(h/100)
            u,v=est['u'][i],est['v'][i]
            uc,vc=est['u_ci'][:,j],est['v_ci'][:,j]
            # Intervals can be asymmetric about their estimate. Draw endpoints
            # directly rather than requiring the mean inside a percentile CI.
            ax.plot(uc,[v,v],color=color,lw=1,alpha=.75)
            ax.plot([u,u],vc,color=color,lw=1,alpha=.75)
            ax.scatter(u,v,color=color,s=22 if h not in (100,500) else 45,
                       marker='o' if h not in (100,500) else '*',zorder=5)
            if h in (0,1000,3000,6000,8000):
                ax.annotate(f"{h/1000:g}",(u,v),xytext=(3,7 if mode=='T' else -14),
                            textcoords='offset points',color=color,fontsize=10)
    for r in (5,10,15):
        ax.add_patch(Circle((0,0),r,fill=False,ls='--',color='.65',lw=.8))
    ax.set(xlim=(-20,17),ylim=(-2,16),xlabel=r"$u$ (m s$^{-1}$)",ylabel=r"$v$ (m s$^{-1}$)")
    ax.set_aspect('equal');ax.grid(alpha=.25)
    ax.axhline(0,color='.4',lw=.7);ax.axvline(0,color='.4',lw=.7)
    ax.legend(loc='upper left',frameon=False,fontsize=11)
    skew.ax.set_xlabel("Temperature (°C)");skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.legend(loc='lower left',fontsize=10,framealpha=.9)
    fig.text(.57,.11,"Shading and crossbars: pointwise 95% confidence intervals\n"
             "Storm-day bootstrap; 52 tornadic and 57 nontornadic cases",fontsize=10)
    out = ROOT/"manuscript/CH2/figures/mean_soundings_hodographs_ci.png"
    fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    audit=ROOT/"manuscript/CH2/revisions/mwr_revision_audit_20261103"
    pd.DataFrame(records).to_csv(audit/"sounding_profile_intervals.csv",index=False)
    np.savez_compressed(audit/"sounding_profile_arrays.npz",height_m=heights,band_height_m=heights[band_idx],**stored)
    (audit/"sounding_figure_metadata.json").write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seed=20261103,resamples=50000,days=len(days),
        grain="case-mean profiles, resampling joint 1200–1200 UTC storm days",
        winds="stored rotated/storm-relative profiles; knots multiplied by 0.514444",
        vertical_grid="10 m AGL for means; 100 m AGL for pointwise intervals",
        intervals="95% percentile, marginal by variable and height; not simultaneous",
        output=str(out.relative_to(ROOT))),indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
