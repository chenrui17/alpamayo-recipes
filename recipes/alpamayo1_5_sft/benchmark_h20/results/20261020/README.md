# 2026-10-20 H20 Stage2 evidence archive

`runs.json` / `runs.csv` index the completed **non-profiled** runs currently
archived. Each `runs/<run_id>/` contains the original all-rank step records,
saved summary, composed and training configurations, manifest, available
`training/stage2*.json` runtime/preload/candidate reports, and `provenance.json`.
Only complete runs with a matching run ID are accepted. Every saved timing mean
was independently recomputed from the raw rank records and compared to its saved
summary. Source Git commit and SHA256 of the actual tracked patch are mandatory;
the old runs' empty patch has the normal SHA256 of empty bytes.
The index also exposes requested optimization switches and explicit Trainer
overrides from the original manifest. A null override means no CLI override was
recorded; consult the composed configuration for its resolved/default value.
`stage2_data.json` is written by rank 0: its preload wall time is one rank's
measurement, not an all-rank mean or sum. Each rank constructs its own cache;
the reported tensor bytes describe that rank's cache footprint.

Each provenance file records both original-source and public-file SHA256 values.
Sanitization can change a public file's bytes: the original `config_sha256` in a
manifest still identifies the source configuration, while `public_sha256` in
provenance identifies the archived version. The numeric timing/loss/memory data
are preserved. Runtime settings differ among historical and optimized runs;
these rows do not imply isolated ablations. New runs can be added while other
experiments continue; the index is the authoritative list for this snapshot.

`annotations.json` records independent activation/review notes without rewriting
the original run files. In particular, `latest_factor_r1` completed its timing
window but had no factor-cache hits, so `candidate_activation=false` and
`comparison_eligible=false`. It is retained as a failed-activation diagnostic and
excluded from both performance figures and evidence for a factor-reuse gain.

Raw archives, `train.log`, patch bodies, and duplicate harness snapshots are not
copied into this public archive. Credentials and connection settings are never
included. The public environment record uses **only an image digest**, not an
internal registry hostname, repository, or tag. The archiver supports exact
credential redaction, private-IP removal and private-root/registry replacement.
An allowlist of result files is used; tar members are read in memory without
extracting arbitrary paths. The source archive remains in the task's untracked
local artifacts area.

`profile/latest_io_profile_r1/` and `profile/latest_combined_profile_r1/` include the independently labeled profile-mode
steps, manifest/configuration, analysis JSON and CSV, and source fingerprints.
Its capture validation reports eight processes, eight devices, and GPU kernels
inside every optimizer step 5–8. Large `.nsys-rep`/SQLite files are retained with
the original task artifacts rather than duplicated into Git. See
[profile_findings.md](profile_findings.md) for the read-only SQL/source audit.
These profile durations are never substituted for the non-profiled timing means.
`profile/comparison.json` contains the read-only SQLite/NVTX comparison, including
per-step crop synchronization counts. Both profiles pass the strict eight-process,
eight-device, 32-step coverage checks. The IO profile reports zero boundary-clipped
kernels; the combined profile reports two, as retained in its original analysis.
`profile/boundary_kernels.json` identifies the two FusedAdamW kernels and the
0.226098 ms total duration outside the selected boundary. See
`profile_findings.md` for the boundary audit. The before/after
profile changes IO/skip-head to the combined Liger + DDP + prefix + factor setup,
so its broad timing changes do not isolate any one optimization.

`checks/correctness_r2.json` is the completed, passed six-phase real-model
comparison (baseline, skip-head, prefix, trajectory, Liger, combined), retaining
all gradient statistics and candidate-activation/cache-hit guards.
`checks/correctness_r2.sources.json` fingerprints the actual runtime sources;
its note explicitly states that fingerprints were collected **after** the run,
with sources unchanged during the run. It is a sidecar and does not rewrite or
pretend to be startup metadata in the original report.
`checks/data_equivalence_r2.json` is the completed, passed comparison of all
20 unique real samples. Each check also has a provenance file with source and
public-file SHA256 values. In-progress reports are not copied into this archive.
`checks/correctness_r3.json` additionally checks baseline, trajectory, Liger and
combined behavior after the factor-cache invalidation fix. It embeds the actual
tracked-patch fingerprint and hashes for 15 source paths (10 Python sources and
five AppleDouble metadata files copied with the source tree). The separate
skip-head/prefix checks in r2 remain historical evidence; r3's combined phase
also exercises prefix behavior.
`checks/ci/` retains the completed CPU logs: 44 candidate tests, 32 upstream
tests, eight Stage1 Alpamayo1.5 tests, eight Stage1 Alpamayo1 tests, and a separate
successful compile check. The final local harness/Nsight unittest log records
another 16 passed tests, giving 108 CPU tests across these recorded suites.
The clean-checkout commit supplied by the task
coordinator is recorded explicitly in provenance; the original status JSON
itself only records completion times and return codes. These checks do not imply
that dependency-lock generation or build CI was run.

The static figures are reproducible from the public data:

```bash
python plot_results.py
```

`timing_means.{png,svg}` shows the mean and p90 of the 16 steady intervals.
`latest_step_intervals.{png,svg}` shows raw per-step maxima across ranks for the
reference configurations and confirmation repeats, exposing variance without
averaging it away. `figures.json` lists the exact runs used in each figure; other
candidate step records remain available in their run folders. The source plotting script uses CPU
Matplotlib; it never starts training or accesses a GPU.
`confirmation_summary.json` equally weights the two independent run means for
baseline, Liger control and combined configurations. Its run-to-run changes
remain visible in the figures; 16 intervals within a run are not treated as 16
independent repetitions, and no confidence interval or cause of drift is claimed.
The standalone prefix, bucket, factor and loader candidates use the
IO + skip-head + Liger configuration as their reference. "Combined" settings
must be read from their manifests rather than inferred from their run names.

To archive additional already-completed runs, use `archive_runs.py --help`.
The optional remote path invokes an existing credential-aware SSH helper solely
to read selected result files and hash the existing patch; it cannot launch a
training command. Existing local sources or compressed archives can be used
without SSH. Credentials supplied for exact-value redaction are not persisted.

`python verify_archive.py` verifies every public-file hash in provenance and
recomputes each timing mean. Optional local-only `--secret-file` and
`--environment-file` inputs also check the exact credential and actual registry
hostname without recording either value. `archive_audit.json` records counts and
the result; it must be regenerated after adding files or updating figures.
