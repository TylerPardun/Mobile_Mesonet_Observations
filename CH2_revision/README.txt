CH2 revision analysis, version 2026-11-03
MWR-D-26-0208: Surface Rear-Flank Downdraft Observations in Supercells
Tyler J. Pardun and coauthors

Contents
- output/run_35685825: 40 group composite fields in 20 NumPy archives,
  pointwise 95% confidence limits, effective storm-day support, raw P values,
  BH/BY adjusted P values, source metadata, and numerical reproduction checks.
- data: per-case sounding profiles and metrics, case/event catalogue, and
  median 3-km radar reflectivity composites used to render the revised figures.
- manuscript/CH2: fixed case reference states, sounding intervals and profile
  arrays, convergence checks, and the returned-results audit.
- scripts: sounding uncertainty, figure rendering, returned-result validation,
  and reference-state selection code.
- schooner: the prespecified protocol, exact pooled Barnes/day-jackknife code,
  tests, configuration, and cluster execution instructions.

Reproduce the figures and validate the returned inference
Use Python 3.11 or newer and the packages listed in requirements.txt.
Run commands from this CH2_revision directory:
  python scripts/mwr_finalize_return.py
  python scripts/mwr_sounding_uncertainty.py
  python scripts/mwr_sounding_figure.py
  python -m unittest discover -s scripts -p 'test_mwr_sounding_uncertainty.py'
  python -m unittest discover -s schooner -p 'test_barnes_uncertainty.py'
Generated figures are written to manuscript/CH2/figures.
The full Barnes calculation uses the observation cache and original grids
specified by schooner/run_revision.py; see schooner/README.txt for execution.
The release files themselves can be checked with python verify_release.py.
Pickle inputs are the project's own Python exports; load them only from a
trusted copy of this repository. Numeric result grids use NumPy NPZ format.

Scientific interpretation
All 40 full-sample scalar grids reproduce to numerical precision. The joint
family contains 34,439 eligible grid cells across four variables and five times.
Neither BY nor BH adjustment yields a detection at q=0.05. This does not imply
that tornado outcomes are equivalent. Eligibility is fixed at ten effective
storm days per group, and the maps are conditional on the original processing,
base states, alignment, and smoothing. The 95% intervals are pointwise.
The protocol was fixed before examining the returned composite results.

Data dictionary
Grid coordinates X and Y are kilometers relative to the mesocyclone, with +X
along storm motion and +Y to its left. pgru is ground-relative wind perturbation
parallel to motion (m/s); thv and the are virtual/equivalent potential temperature
perturbations (K); qv is water vapor mixing-ratio perturbation (g/kg).
T and N identify tornadic and nontornadic composites. ci_low and ci_high are
pointwise 95% limits; eligible indicates supported mean inference; test_eligible
requires both subsets. significant is the prespecified BY q<=0.05 mask.
Vector fields are rotated storm-relative wind components in m/s.
Times are minutes relative to tornadogenesis (T) or peak AzShear03 (N).
Sounding intervals use 50,000 joint 1200-1200 UTC storm-day bootstrap samples,
seed 20261103, with equal case weighting after averaging profiles within cases.
Reference-state CSV units are included in its field names. Its z_ref_m is fixed
per case; cached observation z_ref_m varies by window and is a distinct datum.

Source and provenance
The returned run_manifest.json records source hashes, settings and software.
MANIFEST.json records SHA-256 digests of this version's files.
