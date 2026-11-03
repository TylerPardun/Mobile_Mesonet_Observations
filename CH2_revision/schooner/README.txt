MWR-D-26-0208 — Schooner revision analysis package
Prepared 3 November 2026

Purpose
-------
Address the editor's requests for uncertainty on composite means and multiplicity-
controlled significance of the tornadic-minus-nontornadic maps. The package reads
existing large files in place. It does not rebuild or alter raw data, change the
base states, or overwrite the manuscript's figures.

RUN ON SCHOONER
--------------
1. Extract this package somewhere in your condo space, then enter its directory.
2. Activate the Python environment used for the mobile-mesonet analysis. It needs
   numpy, scipy, pandas, matplotlib, and openpyxl. No SHARPpy calculations are run.
3. Submit:

   bash submit.sh /condo/swat_z6/tpardun/phd_stuff

The helper uses the current environment's Python. To select another interpreter:

   export MWR_PYTHON=/absolute/path/to/environment/bin/python
   bash submit.sh /condo/swat_z6/tpardun/phd_stuff

The job requests one node, 20 CPUs, 28 GB, and 24 hours on 32gb_20core. It runs one
process with threaded numerical libraries to avoid multiplying the large input
cache in memory. The helper checks the partition against live sinfo output. If
running on the new Sooner queues instead, explicitly set MWR_PARTITION to the
appropriate visible partition (for example sooner_test_32gb_20core).

OSCER documentation, checked 3 November 2026:
https://www.ou.edu/oscer/using-the-cluster/partitions
https://www.ou.edu/oscer/using-the-cluster/submitting-jobs
https://www.ou.edu/oscer/getting-started/hpc-intro

The first source lists 32gb_20core, up to ~30 GB/node and a 48-hour maximum. The
24-hour request is a cap, not a measured runtime prediction. Run the resampling
job through Slurm. Inventory checks below are suitable for a login node.

OPTIONAL INPUT CHECK
--------------------
   python run_revision.py --inventory \
       --root /condo/swat_z6/tpardun/phd_stuff \
       --output results/inventory_01

This checks file locations and small metadata without loading the observation
cache or doing objective analyses. Use a new output directory for each run.
The discovery searches only three directory levels and excludes large radar/raw
observation directories. Ambiguous paths fail explicitly. If necessary, pass
--pickle /path/monte_carlo_stuff.pkl, --barnes /path/barnes_precomputed, and
--transects /path/transect_times.xlsx to run_revision.py in job.sbatch.

Required files:
- monte_carlo_stuff.pkl: the existing [mesonet_data, sounding_data, ...] export;
- transect_times.xlsx: the corresponding case catalogue;
- barnes_precomputed/{pgru,thv,the,qv}/{T,N}_{offset:+03d}.pkl for offsets
  -15, -5, 0, +5 and +15 minutes;
- preferably barnes_precomputed/metadata.json and/or metadata.pkl.

The full mobile_mesonet_data_tsc_5_min.pkl is NOT read. The package guards against
loading a composite cache larger than 8 GB on this memory allocation. If that
check fails, return the inventory for a targeted extraction workflow; do not
substitute a larger unverified file or run it on a login node.

WHAT IT DOES
------------
1. Records file hashes, configuration, case/day counts, embedded base states,
   and copies small provenance files. Checks case identity and t0 alignment.
2. Reads actual archived Barnes metadata where available. Configured defaults
   are 6.7 km^2, gamma=0.3, and minimum one neighbor, from the prior archived
   product inventory. These differ from the manuscript's minimum of three and
   from dormant plotting code's kappa=5.46. Missing metadata is flagged for review.
3. Recomputes every saved scalar grid. All finite masks and values must agree
   within the documented floating-point tolerance. A mismatch stops inference.
4. Recalculates both Barnes passes after deleting each 1200–1200 UTC storm day.
   The same day is removed from both groups, preserving their shared-day
   covariance. Original pooled-observation weighting, QC mask, base states,
   geometry, grid, and smoothing settings are retained.
5. Produces approximate pointwise 95% jackknife t intervals and two-sided tests
   of the difference. Cells require at least 10 effective storm days in BOTH
   groups and finite results for all day deletions. Effective days are the
   inverse concentration of first-pass Barnes weights aggregated by day.
6. Applies Benjamini–Yekutieli FDR at q=0.05 across ALL eligible cells in all four
   fields and five times as one family. This accommodates arbitrary dependence
   among the cell tests, subject to validity of the underlying approximate
   p-values. Benjamini–Hochberg results are also returned as a sensitivity check.
7. Renders candidate figures, confidence-width maps and a small return archive.
   Thin contours give pointwise CI half-widths. Black dots show primary BY FDR
   detections; gray overlays show insufficient support. Saved SR vectors and
   available 3-km reflectivity provide context. Pointwise intervals are not
   simultaneous confidence bands. No pointwise p<0.05 stippling is substituted
   if the corrected analysis finds no detections.

RETURN THIS FILE
----------------
   results/run_<job-number>.tar.gz

The job prints its exact location. Return the archive plus mwr_revision_<job-number>.log
if the job fails or is killed. A caught error still creates a diagnostic archive.
A scheduler kill or failed dependency/unit-test check may only leave the log.
Do not transfer the large source pickles.

The archive contains:
- run_manifest.json and grid_reproduction.csv;
- cache_case_inventory.csv and embedded_base_states.csv;
- compact .npz grids with means, CIs, support, p-values and adjusted q-values;
- significance_summary.csv and candidate figures/;
- small provenance/ files needed to settle the methods description;
- error.txt and mismatch grids if an archived product did not reproduce.

STOPPING POINT BEFORE RESUBMISSION
---------------------------------
A successful run means the analysis completed. It does not mean the manuscript
is ready to submit. Return the archive so the actual results can be checked,
the base-state and Barnes-method discrepancies resolved, claims/captions updated,
figures finalized, and the editor response and final line count completed.
The present main.tex already contains the local editorial and sounding-uncertainty
revisions. Keep using that file; do not replace it with a copy from this package.

VALIDATION AND LIMITS
--------------------
See PROTOCOL.txt and VALIDATION.txt. The numerical core is checked against an
independent slow implementation for the full sample and every cluster deletion.
A complete synthetic run and a deliberate archive-mismatch run were exercised
locally. The actual Schooner dataset has NOT been run or benchmarked here.
The intervals quantify sampling uncertainty conditional on fixed processing.
They do not quantify instrument error, base-state uncertainty, feature-registration
error, or the representativeness of the targeted storm sample.
