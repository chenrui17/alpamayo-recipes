"""Regenerate static figures from the public archived timing records (CPU only)."""

import json
import os
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".font-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "legacy_baseline_r3": "Legacy baseline",
    "legacy_io_skip_r3": "Legacy IO + skip-head",
    "latest_baseline_r3": "Upstream baseline (run 1)",
    "latest_io_skip_r4": "Latest IO + skip-head",
    "latest_liger_r1": "IO + skip-head + Liger (run 1)",
    "latest_ddp128_r1": "IO + Liger + DDP 128 MiB",
    "latest_ddp256_r1": "IO + Liger + DDP 256 MiB",
    "latest_prefix_r1": "IO + Liger + prefix/boundary reuse",
    "latest_nopreload_r1": "IO + Liger, no preload",
    "latest_workers0_r1": "IO + Liger, workers 0",
    "latest_factor_r2": "IO + Liger + factor reuse",
    "legacy_liger_r1": "Legacy IO + skip-head + Liger",
    "latest_liger_r2": "IO + skip-head + Liger (run 2)",
    "latest_combined_r1": "Combined (run 1)",
    "latest_combined_r2": "Combined (run 2)",
    "latest_baseline_r4": "Upstream baseline (run 2)",
}
ORDER = [
    "legacy_baseline_r3", "legacy_io_skip_r3", "legacy_liger_r1",
    "latest_baseline_r3", "latest_baseline_r4", "latest_io_skip_r4",
    "latest_liger_r1", "latest_liger_r2", "latest_ddp128_r1", "latest_ddp256_r1",
    "latest_prefix_r1", "latest_factor_r2", "latest_nopreload_r1", "latest_workers0_r1",
    "latest_combined_r1", "latest_combined_r2",
]
STEP_PLOT_RUNS = {
    "latest_baseline_r3", "latest_baseline_r4",
    "latest_liger_r1", "latest_liger_r2",
    "latest_combined_r1", "latest_combined_r2",
}


def save(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = json.loads((ROOT / "runs.json").read_text())["runs"]
    rows = [row for row in rows if row.get("comparison_eligible", True)]
    rows.sort(key=lambda row: (ORDER.index(row["run_id"]) if row["run_id"] in ORDER else len(ORDER), row["run_id"]))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    figure, axes = plt.subplots(figsize=(9.0, max(4.8, len(rows) * 0.50 + 1.8)))
    positions = list(range(len(rows)))
    colors = ["#8a919d" if "baseline" in row["run_id"] else "#3973ad" if "io_skip" in row["run_id"] else "#32a394" for row in rows]
    axes.barh(positions, [row["mean_seconds"] for row in rows], color=colors, height=0.62)
    axes.scatter([row["p90_seconds"] for row in rows], positions, marker="D", s=25, color="#d46b2a", label="p90", zorder=3)
    axes.set_yticks(positions, [LABELS.get(row["run_id"], row["run_id"]) for row in rows])
    axes.invert_yaxis()
    for index, row in enumerate(rows):
        axes.text(row["mean_seconds"] - 0.025, index, f'{row["mean_seconds"]:.4f}', ha="right",
                  va="center", fontsize=9, color="white", weight="semibold")
    axes.set_xlabel("Seconds per optimizer step (lower is better)")
    axes.set_title("H20 Stage2: 8 GPUs, microbatch 1, accumulation 1", loc="left", pad=16)
    axes.set_xlim(0, max(row["p90_seconds"] for row in rows) * 1.12)
    axes.grid(axis="x", color="#e4e7eb", linewidth=0.7)
    axes.set_axisbelow(True)
    axes.legend(loc="lower right", frameon=False)
    figure.subplots_adjust(bottom=0.11)
    figure.text(0.02, 0.015, "Non-profiled steps 5–20; each interval is the maximum across ranks.\n"
                "Bars: mean of 16 intervals. Historical configurations differ; see run manifests.", fontsize=9, color="#505866")
    save(figure, "timing_means")

    latest = [row for row in rows if row["run_id"] in STEP_PLOT_RUNS]
    figure, axes = plt.subplots(figsize=(9.2, 4.9))
    for row in latest:
        raw = json.loads((ROOT / "runs" / row["run_id"] / "steps.json").read_text())
        values = [{record["step"]: record["step_seconds"] for record in rank["records"]
                   if record.get("step_seconds") is not None} for rank in raw["ranks"]]
        steps = list(range(raw["steady_start_step"], raw["steady_end_step"] + 1))
        maxima = [max(rank[step] for rank in values) for step in steps]
        group = "baseline" if "baseline" in row["run_id"] else "liger" if "liger" in row["run_id"] else "combined"
        repeat = row["run_id"] in {"latest_baseline_r4", "latest_liger_r2", "latest_combined_r2"}
        axes.plot(steps, maxima, marker="^" if repeat else "o", markersize=3.5,
                  linestyle="--" if repeat else "-", color={"baseline": "#747c89", "liger": "#3973ad", "combined": "#228a71"}[group],
                  linewidth=1.4, label=LABELS.get(row["run_id"], row["run_id"]))
    axes.set_xticks(range(5, 21))
    axes.set_xlabel("Optimizer step")
    axes.set_ylabel("Maximum rank interval (seconds)")
    axes.set_title("Latest code: reference and confirmation step intervals", loc="left", pad=14)
    axes.grid(color="#e4e7eb", linewidth=0.7)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    save(figure, "latest_step_intervals")
    (ROOT / "figures.json").write_text(json.dumps({
        "timing_means": [row["run_id"] for row in rows],
        "latest_step_intervals": [row["run_id"] for row in latest],
        "excluded_failed_activation_runs": [row["run_id"] for row in json.loads((ROOT / "runs.json").read_text())["runs"] if not row.get("comparison_eligible", True)],
        "source": "runs.json plus runs/<run_id>/steps.json", "profile_timings_used": False,
    }, indent=2) + "\n")
    indexed = {row["run_id"]: row for row in rows}
    groups = {
        "upstream_baseline": ["latest_baseline_r3", "latest_baseline_r4"],
        "liger_control": ["latest_liger_r1", "latest_liger_r2"],
        "combined": ["latest_combined_r1", "latest_combined_r2"],
    }
    if all(run_id in indexed for run_ids in groups.values() for run_id in run_ids):
        descriptions = {}
        for group, run_ids in groups.items():
            means = [indexed[run_id]["mean_seconds"] for run_id in run_ids]
            descriptions[group] = {
                "run_ids": run_ids, "independent_runs": len(run_ids), "steady_intervals_per_run": 16,
                "run_mean_seconds": means, "mean_of_run_means_seconds": statistics.mean(means),
                "second_run_change_relative_to_first": means[1] / means[0] - 1,
            }
        combined = descriptions["combined"]["mean_of_run_means_seconds"]
        comparisons = {}
        for group in ("upstream_baseline", "liger_control"):
            reference = descriptions[group]["mean_of_run_means_seconds"]
            comparisons[group + "_to_combined"] = {
                "step_time_reduction_fraction": 1 - combined / reference,
                "throughput_ratio": reference / combined,
            }
        (ROOT / "confirmation_summary.json").write_text(json.dumps({
            "groups": descriptions, "comparisons": comparisons,
            "note": "Descriptive results from two independent runs per configuration; each has 16 synchronized rank-max intervals. Step intervals are not independent repetitions. No confidence interval or drift-cause attribution is claimed.",
        }, indent=2) + "\n")
    cache = ROOT / ".matplotlib-cache"
    if os.environ.get("MPLCONFIGDIR") == str(cache) and cache.exists():
        shutil.rmtree(cache)
    cache = ROOT / ".font-cache"
    if os.environ.get("XDG_CACHE_HOME") == str(cache) and cache.exists():
        shutil.rmtree(cache)
    print("Generated timing_means.{png,svg} and latest_step_intervals.{png,svg}")


if __name__ == "__main__":
    main()
