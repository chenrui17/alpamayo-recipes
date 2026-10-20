# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an Nsight Systems SQLite export; write JSON and optional flat CSV tables.

Uses only the standard library and never modifies the input database. Timestamps
and durations are nanoseconds. See --help for capture/export commands and limits.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


Interval = tuple[int, int]
STEP_RE = re.compile(r"(?:^|[^a-z0-9])optimizer[_ /-]*step[_ :=/-]*(\d+)(?:$|\D)", re.I)
PHASE_PATTERNS = [
    ("visual", r"visual|vision"), ("language", r"language|llm|text_model"),
    ("action", r"action"), ("trajectory", r"trajectory|traj(?:_|\b)"),
    ("expert", r"expert"), ("head", r"head"), ("vlm", r"vlm"),
]
DATA_RE = re.compile(r"dataloader|collat|next[_ .:/-]*batch|prepare[_ .:/-]*input|input[_ .:/-]*wait|queue[_ .:/-]*(?:get|wait)", re.I)
SOURCE_DOCS = {
    "collection": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html",
    "sqlite_schema": "https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html",
    "multigpu_profiler_stop": "https://forums.developer.nvidia.com/t/nsys-profile-can-hang-for-a-long-time-when-profiling-pytorch-distributed-training-runs/340759/5",
}


def merge(intervals: list[Interval]) -> list[Interval]:
    output: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def duration(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in merge(intervals))


def intersections(interval: Interval, windows: list[Interval]) -> list[Interval]:
    start, end = interval
    return [(max(start, a), min(end, b)) for a, b in windows if max(start, a) < min(end, b)]


def subtract(windows: list[Interval], occupied: list[Interval]) -> list[Interval]:
    occupied = merge(occupied)
    output = []
    for start, end in merge(windows):
        cursor = start
        for a, b in occupied:
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                output.append((cursor, min(a, end)))
            cursor = max(cursor, min(b, end))
        if cursor < end:
            output.append((cursor, end))
    return output


def kernel_group(name: str) -> str:
    name = name.lower()
    if re.search(r"nccl|msccl", name):
        return "NCCL"
    if re.search(r"flash|fmha|fused[_: ]*attention|attention[_: ]*kernel|scaled_dot_product", name):
        return "FA"
    if re.search(r"gemm|cublas|cutlass|matmul|(?:^|[_:< ])mma(?:[_:< ]|$)", name):
        return "GEMM"
    if re.search(r"elementwise|pointwise|vectorized[_: ]|unrolled[_: ]", name):
        return "pointwise"
    return "other"


def phase_name(text: str) -> str | None:
    for phase, pattern in PHASE_PATTERNS:
        if re.search(pattern, text, re.I):
            return phase
    return None


def api_category(name: str) -> str:
    name = name.lower()
    if "synchronize" in name:
        return "explicit_sync"
    if "waitevent" in name or "waitvalue" in name:
        return "device_wait_enqueue"
    if "memcpy" in name:
        return "async_memcpy_enqueue" if "async" in name else "potentially_blocking_memcpy"
    if "malloc" in name or "memalloc" in name or "free" in name:
        return "async_allocation_or_free" if "async" in name else "potentially_synchronizing_allocation_or_free"
    if "launch" in name:
        return "launch"
    return "other"


def process_key(row: dict[str, Any]) -> str:
    for name in ("globalpid", "globaltid"):
        if row.get(name) is not None:
            return str(int(row[name]) >> 24)
    if row.get("pid") is not None:
        return str(int(row["pid"]))
    return "unknown"


def process_info(key: str, rank_pids: dict[int, int]) -> dict[str, Any]:
    pid = int(key) & 0xFFFFFF if key != "unknown" else None
    return {"process_key": key, "pid": pid, "rank": rank_pids.get(pid)}


class Export:
    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        self.tables = {
            row[0].lower(): row[0] for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        self.schema: dict[str, list[str]] = {}
        for table in self.tables.values():
            self.schema[table] = [row[1] for row in self.connection.execute("PRAGMA table_info(" + self.quote(table) + ")")]
        self.strings = {
            int(row["id"]): str(row["value"]) for row in self.rows("StringIds")
            if row.get("id") is not None and row.get("value") is not None
        }

    @staticmethod
    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def rows(self, table: str) -> list[dict[str, Any]]:
        actual = self.tables.get(table.lower())
        if actual is None:
            return []
        return [{key.lower(): value for key, value in dict(row).items()}
                for row in self.connection.execute("SELECT * FROM " + self.quote(actual))]

    def name(self, row: dict[str, Any], nvtx: bool = False) -> str:
        # NVTX literal text is not a StringIds identifier. Registered strings use
        # textId; COALESCE(text, StringIds.value) is the relevant schema relation.
        if nvtx:
            if row.get("text") is not None and str(row["text"]):
                return str(row["text"])
            value = row.get("textid")
            return self.strings.get(int(value), f"<unresolved textId={value}>") if value is not None else "<unnamed>"
        for field in ("demangledname", "shortname", "nameid", "mangledname", "name", "text"):
            value = row.get(field)
            if value is not None:
                return self.strings.get(value, str(value))
        return "<unnamed>"

    def events(self, table: str, nvtx: bool = False) -> list[dict[str, Any]]:
        output = []
        for row in self.rows(table):
            if row.get("start") is None:
                continue
            row.update({"table": self.tables[table.lower()], "name": self.name(row, nvtx),
                        "process_key": process_key(row)})
            output.append(row)
        return output


def valid_interval(event: dict[str, Any]) -> Interval | None:
    if event.get("end") is None:
        return None
    start, end = int(event["start"]), int(event["end"])
    return (start, end) if end > start else None


def analyze(path: Path, expected_ranks: int | None = None, first_step: int = 5, last_step: int = 8,
            rank_pids: dict[int, int] | None = None, top: int = 20) -> dict[str, Any]:
    if first_step < 1 or last_step < first_step:
        raise ValueError("Invalid requested optimizer-step window")
    export = Export(path)
    try:
        return _analyze(export, expected_ranks, first_step, last_step, rank_pids or {}, top)
    finally:
        export.connection.close()


def _analyze(export: Export, expected_ranks: int | None, first_step: int, last_step: int,
             rank_pids: dict[int, int], top: int) -> dict[str, Any]:
    warnings = []
    nvtx = export.events("NVTX_EVENTS", nvtx=True)
    kernels = export.events("CUPTI_ACTIVITY_KIND_KERNEL")
    if not kernels:
        # Some old exporters use the concurrent-kernel activity name instead.
        kernels = export.events("CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
    apis = [event for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
            for event in export.events(table)]
    osrt = [event for key, table in export.tables.items() if key in {"osrt_api", "osrt_events"}
            for event in export.events(table)]
    memory = [event for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
              for event in export.events(table)]
    if not kernels:
        warnings.append("No GPU kernel events were exported; GPU timing evidence is unavailable.")
    if not nvtx:
        warnings.append("NVTX_EVENTS is absent or empty; requested step window cannot be verified.")
    if not osrt:
        warnings.append("No OS runtime events; CPU input-wait attribution needs NVTX or another trace.")
    step_events = []
    for event in nvtx:
        match = STEP_RE.search(event["name"])
        if match:
            step_events.append({**event, "step": int(match.group(1))})
    by_process = defaultdict(list)
    for event in step_events:
        by_process[event["process_key"]].append(event)
    expected_steps = set(range(first_step, last_step + 1))
    process_windows: dict[str, list[Interval]] = {}
    validation_processes = []
    step_rows = []
    interstep_gaps = []
    for key, events in sorted(by_process.items()):
        completed = defaultdict(list)
        open_steps = []
        for event in events:
            interval = valid_interval(event)
            if interval:
                completed[event["step"]].append(interval)
                step_rows.append({**process_info(key, rank_pids), "step": event["step"],
                                  "start_ns": interval[0], "end_ns": interval[1], "duration_ns": interval[1] - interval[0],
                                  "thread": str(event.get("globaltid", "unknown"))})
            else:
                open_steps.append(event["step"])
        selected = [interval for step in expected_steps for interval in completed.get(step, [])]
        if selected:
            process_windows[key] = [(min(a for a, _ in selected), max(b for _, b in selected))]
        missing = sorted(expected_steps - set(completed))
        unexpected = sorted(set(completed) - expected_steps)
        duplicate = sorted(step for step, values in completed.items() if len(values) != 1)
        validation_processes.append({**process_info(key, rank_pids), "observed_steps": sorted(completed),
                                     "missing_steps": missing, "unexpected_steps": unexpected,
                                     "duplicate_steps": duplicate, "open_or_invalid_steps": sorted(open_steps),
                                     "valid": not (missing or unexpected or duplicate or open_steps)})
        for step in range(first_step, last_step):
            if len(completed.get(step, [])) == len(completed.get(step + 1, [])) == 1:
                start, end = completed[step][0][1], completed[step + 1][0][0]
                if end > start:
                    interstep_gaps.append({**process_info(key, rank_pids), "after_step": step,
                                           "before_step": step + 1, "start_ns": start, "end_ns": end,
                                           "gap_ns": end - start})
    all_events = [*nvtx, *kernels, *apis, *osrt, *memory]
    all_intervals = [value for event in all_events if (value := valid_interval(event))]
    trace_bounds = (min(a for a, _ in all_intervals), max(b for _, b in all_intervals)) if all_intervals else None
    selected_windows = merge([interval for windows in process_windows.values() for interval in windows])
    if not selected_windows and trace_bounds:
        selected_windows = [trace_bounds]
        warnings.append("Using all exported event bounds because no complete requested step ranges were found.")
    unmapped = defaultdict(int)

    def selected(event: dict[str, Any]) -> list[Interval]:
        interval = valid_interval(event)
        if interval is None:
            return []
        key = event["process_key"]
        if key == "unknown":
            unmapped[event["table"]] += 1
            return intersections(interval, selected_windows)
        if process_windows and key not in process_windows:
            return []
        return intersections(interval, process_windows.get(key, selected_windows))

    # NVTX coarse CPU phase intervals, independently unioned on each thread.
    phase_events = []
    for event in nvtx:
        phase = phase_name(event["name"])
        if phase is not None and (interval := valid_interval(event)):
            phase_events.append({**event, "phase": phase, "interval": interval})
    phase_intervals = defaultdict(list)
    phase_examples = defaultdict(set)
    per_thread = defaultdict(list)
    for event in phase_events:
        thread = str(event.get("globaltid", "unknown"))
        key = (event["process_key"], thread)
        per_thread[key].append(event)
        for a, b in selected(event):
            phase_intervals[(*key, event["phase"])].append((a, b))
        phase_examples[(event["process_key"], event["phase"])].add(event["name"])
    exclusive = defaultdict(int)
    for key, events in per_thread.items():
        points = []
        for index, event in enumerate(events):
            for a, b in selected(event):
                points.extend([(a, 1, index), (b, -1, index)])
        active = set()
        previous = None
        for timestamp, operation, index in sorted(points):
            if previous is not None and timestamp > previous and active:
                innermost = max(active, key=lambda i: (events[i]["interval"][0], -events[i]["interval"][1], i))
                exclusive[(*key, events[innermost]["phase"])] += timestamp - previous
            if operation == 1:
                active.add(index)
            else:
                active.discard(index)
            previous = timestamp
    nvtx_rows = []
    for (key, thread, phase), intervals in sorted(phase_intervals.items()):
        nvtx_rows.append({**process_info(key, rank_pids), "thread": thread, "phase": phase,
                          "range_count": len(intervals), "inclusive_thread_union_ns": duration(intervals),
                          "exclusive_recognized_phase_ns": exclusive[(key, thread, phase)],
                          "example_names": sorted(phase_examples[(key, phase)])[:5]})
    phase_lookup = {}
    for key, events in per_thread.items():
        events.sort(key=lambda event: event["interval"][0])
        phase_lookup[key] = ([event["interval"][0] for event in events], events)
    launches = defaultdict(list)
    for api in apis:
        if api.get("correlationid") is not None:
            launches[(api["process_key"], int(api["correlationid"]))].append(api)

    def kernel_phase(event: dict[str, Any]) -> str:
        matches = launches.get((event["process_key"], event.get("correlationid")), [])
        matches = sorted(matches, key=lambda api: "RUNTIME" not in api["table"].upper())
        for api in matches:
            key = (api["process_key"], str(api.get("globaltid", "unknown")))
            if key not in phase_lookup:
                continue
            starts, ranges = phase_lookup[key]
            timestamp = int(api["start"])
            for parent in reversed(ranges[:bisect.bisect_right(starts, timestamp)]):
                if parent["interval"][1] >= int(api.get("end") or timestamp):
                    return parent["phase"]
        return "unattributed"

    kernel_buckets = defaultdict(list)
    kernel_names = defaultdict(list)
    kernel_phases = defaultdict(list)
    device_kernels = defaultdict(list)
    device_activity = defaultdict(list)
    device_processes = defaultdict(set)
    kernel_processes = set()
    kernels_per_process = defaultdict(list)
    clipped_kernels = 0
    for event in kernels:
        intervals = selected(event)
        if not intervals:
            continue
        device = str(event.get("deviceid", "unknown"))
        device_processes[device].add(event["process_key"])
        kernel_processes.add(event["process_key"])
        group = kernel_group(event["name"])
        phase = kernel_phase(event)
        if sum(b - a for a, b in intervals) != event["end"] - event["start"]:
            clipped_kernels += 1
        for interval in intervals:
            kernel_buckets[(device, group)].append(interval)
            kernel_names[(device, group, event["name"])].append(interval)
            kernel_phases[(device, phase)].append(interval)
            device_kernels[device].append(interval)
            device_activity[device].append(interval)
            kernels_per_process[event["process_key"]].append((interval, device))
    for event in memory:
        device = str(event.get("deviceid", "unknown"))
        for interval in selected(event):
            device_activity[device].append(interval)
            device_processes[device].add(event["process_key"])
    group_rows = [{"device": device, "group": group, "kernel_count": len(intervals),
                   "sum_kernel_duration_ns": sum(b - a for a, b in intervals),
                   "within_group_busy_union_ns": duration(intervals)}
                  for (device, group), intervals in sorted(kernel_buckets.items())]
    phase_kernel_rows = [{"device": device, "phase": phase, "kernel_count": len(intervals),
                          "sum_kernel_duration_ns": sum(b - a for a, b in intervals)}
                         for (device, phase), intervals in sorted(kernel_phases.items())]
    top_kernels = [{"device": device, "group": group, "name": name, "kernel_count": len(intervals),
                    "sum_kernel_duration_ns": sum(b - a for a, b in intervals)}
                   for (device, group, name), intervals in kernel_names.items()]
    top_kernels.sort(key=lambda row: row["sum_kernel_duration_ns"], reverse=True)
    devices = []
    idle_by_process = defaultdict(list)
    for device, activity in sorted(device_activity.items()):
        keys = device_processes[device]
        windows = merge([interval for key in keys for interval in process_windows.get(key, selected_windows)])
        gaps = subtract(windows, activity)
        for key in keys:
            idle_by_process[key].extend(gaps)
        devices.append({"device": device, "processes": [process_info(key, rank_pids) for key in sorted(keys)],
                        "window_union_ns": duration(windows), "kernel_busy_union_ns": duration(device_kernels[device]),
                        "gpu_activity_busy_union_ns": duration(activity), "no_recorded_gpu_activity_ns": duration(gaps),
                        "largest_no_activity_gaps": [{"start_ns": a, "end_ns": b, "duration_ns": b - a}
                                                     for a, b in sorted(gaps, key=lambda pair: pair[1] - pair[0], reverse=True)[:top]]})
    for gap in interstep_gaps:
        idle = idle_by_process[gap["process_key"]]
        gap["no_recorded_gpu_activity_ns"] = duration(intersections((gap["start_ns"], gap["end_ns"]), merge(idle)))
    steps_without_kernels = []
    for step in step_rows:
        interval = (step["start_ns"], step["end_ns"])
        matches = [(part, device) for kernel, device in kernels_per_process[step["process_key"]]
                   for part in intersections(kernel, [interval])]
        step["overlapping_gpu_kernel_count"] = len(matches)
        step["sum_overlapping_gpu_kernel_duration_ns"] = sum(b - a for (a, b), _ in matches)
        step["devices"] = sorted({device for _, device in matches})
        if first_step <= step["step"] <= last_step and not matches:
            steps_without_kernels.append({**process_info(step["process_key"], rank_pids), "step": step["step"]})

    api_buckets = defaultdict(list)
    api_process_categories = defaultdict(list)
    for event in apis:
        category = api_category(event["name"])
        for interval in selected(event):
            api_buckets[(event["table"], event["process_key"], str(event.get("globaltid", "unknown")), category, event["name"])].append(interval)
            api_process_categories[(event["process_key"], category)].append(interval)
    api_rows = [{"table": table, **process_info(key, rank_pids), "thread": thread,
                 "category": category, "name": name, "count": len(intervals),
                 "sum_host_api_duration_ns": sum(b - a for a, b in intervals),
                 "host_api_union_ns": duration(intervals), "max_host_api_duration_ns": max(b - a for a, b in intervals)}
                for (table, key, thread, category, name), intervals in api_buckets.items()]
    api_rows.sort(key=lambda row: row["sum_host_api_duration_ns"], reverse=True)
    api_category_rows = [{**process_info(key, rank_pids), "category": category,
                          "host_api_union_across_tables_threads_ns": duration(intervals)}
                         for (key, category), intervals in sorted(api_process_categories.items())]
    wait_buckets = defaultdict(list)
    for event in osrt:
        name = event["name"]
        if not re.search(r"poll|select|read|futex|pthread_cond|pthread_mutex|sem_wait|sleep|waitpid", name, re.I):
            continue
        key = (event["process_key"], str(event.get("globaltid", "unknown")), name)
        wait_buckets[key].extend(selected(event))
    wait_rows = []
    for (key, thread, name), intervals in wait_buckets.items():
        idle = merge(idle_by_process[key])
        overlap = [part for interval in intervals for part in intersections(interval, idle)]
        wait_rows.append({**process_info(key, rank_pids), "thread": thread, "name": name,
                          "count": len(intervals), "host_wait_union_ns": duration(intervals),
                          "overlap_with_process_gpu_idle_union_ns": duration(overlap)})
    wait_rows.sort(key=lambda row: row["host_wait_union_ns"], reverse=True)
    data_ranges = [{**process_info(event["process_key"], rank_pids), "name": event["name"],
                    "start_ns": a, "end_ns": b, "duration_ns": b - a}
                   for event in nvtx if DATA_RE.search(event["name"]) for a, b in selected(event)]
    gpu_missing_processes = sorted(set(process_windows) - kernel_processes)
    valid = bool(validation_processes) and all(row["valid"] for row in validation_processes)
    valid = valid and not gpu_missing_processes and not steps_without_kernels and "unknown" not in by_process
    if expected_ranks is not None:
        valid = valid and len(validation_processes) == expected_ranks
    if rank_pids:
        valid = valid and {row["pid"] for row in validation_processes} == set(rank_pids)
    if not phase_events:
        warnings.append("No recognized coarse NVTX phases. Their timings cannot be inferred from kernel names.")
    if unmapped:
        warnings.append("Some event tables lack usable process IDs; those events use the global selected window.")
    if not valid:
        warnings.append("Capture window validation failed; inspect missing/extra/open step labels and missing GPU processes before interpreting results.")
    return {
        "schema_version": 1, "input_sqlite": str(export.path), "time_unit": "nanoseconds",
        "source_docs": SOURCE_DOCS, "sqlite_schema": export.schema,
        "capture_validation": {"valid": valid, "requested_steps": [first_step, last_step],
                               "expected_rank_processes": expected_ranks, "observed_rank_processes": len(validation_processes),
                               "processes": validation_processes, "step_processes_without_gpu_kernels": gpu_missing_processes,
                               "steps_without_gpu_kernels": steps_without_kernels,
                               "export_event_bounds_ns": list(trace_bounds) if trace_bounds else None,
                               "selected_window_union_ns": duration(selected_windows), "clipped_kernel_count": clipped_kernels},
        "optimizer_steps": sorted(step_rows, key=lambda row: (row["process_key"], row["step"])),
        "devices": devices, "kernels_by_device_group": group_rows,
        "kernels_by_device_phase": phase_kernel_rows, "top_kernels": top_kernels[:top],
        "cuda_api": api_rows, "cuda_api_category_unions": api_category_rows,
        "nvtx_phases": nvtx_rows,
        "cpu_data_wait_clues": {"interstep_gaps": interstep_gaps, "nvtx_data_ranges": data_ranges,
                                "osrt_wait_calls": wait_rows[:top]},
        "warnings": warnings,
        "interpretation": [
            "Kernel sums are actual GPU execution durations but overlap across streams/devices; never use the eight-GPU sum as a single-step critical path.",
            "Device busy unions count simultaneous kernels once. GPU activity unions also include exported memcpy/memset; idle gaps are missing recorded activity, not a measured input-wait cause.",
            "NVTX phase inclusive unions overlap across parent/child phases. Exclusive times assign same-thread segments to the innermost recognized phase; cross-thread values still are not wall time.",
            "GPU phase attribution uses process-qualified CUDA correlation IDs and the innermost same-thread phase containing the CPU launch API, not GPU timestamp containment.",
            "Kernel groups and NVTX phase names use conservative name heuristics. Inspect names and unattributed/other groups before drawing conclusions.",
            "CUDA API times are CPU call durations. Runtime and driver calls may nest; categories can overlap. Device wait enqueue APIs do not prove host blocking. Explicit sync includes benchmark callback synchronization.",
            "OSRT waits may reflect dataloading, communication, background threads, or ordinary idle time; GPU-idle overlap and interstep gaps are clues, not proof of a data bottleneck.",
            "PID and global process key identify processes; a CUDA device ID is not assumed to be a distributed rank. Supply --rank-pid for verified rank labels.",
            "This profiled window is diagnostic only. Use separate non-profiled synchronized steps 5-20 for throughput/speedup conclusions.",
        ],
    }


def write_csv_tables(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tables = {key: result[key] for key in ("optimizer_steps", "devices", "kernels_by_device_group", "kernels_by_device_phase",
                                          "top_kernels", "cuda_api", "cuda_api_category_unions", "nvtx_phases")}
    tables.update(result["cpu_data_wait_clues"])
    for name, rows in tables.items():
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / (name + ".csv")).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                              for key, value in row.items()} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="""
Export: nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep
Stats: nsys stats --report=cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,osrt_sum --format=csv --output=stats profile.nsys-rep
Profile a single-node torchrun launcher from outside: nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn --sample=none --cpuctxsw=none --wait=all --capture-range=cudaProfilerApi --capture-range-end=stop --flush-on-cudaprofilerstop=false --output=profile python -m torch.distributed.run ...
Use a unique output prefix and verify every child process has completed optimizer_step_5..8 ranges. --trace-fork-before-exec=true is not needed to follow exec'd torchrun workers and can deadlock fork-only workers. Check nsys profile --help on the installed version. Nsight stats CSV aggregates alone cannot validate step windows or remove overlap; this helper reads the timeline SQLite and exports its own CSV tables.
""")
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--first-step", type=int, default=5)
    parser.add_argument("--last-step", type=int, default=8)
    parser.add_argument("--rank-pid", action="append", default=[], metavar="RANK=PID")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--strict-window", action="store_true", help="Exit 2 after writing results if rank/step/kernel coverage validation fails")
    args = parser.parse_args()
    rank_pids = {}
    try:
        for item in args.rank_pid:
            rank, pid = (int(value) for value in item.split("=", 1))
            if rank < 0 or pid <= 0 or pid in rank_pids or rank in rank_pids.values():
                raise ValueError("Require distinct nonnegative ranks and distinct positive PIDs")
            rank_pids[pid] = rank
        if args.expected_ranks is not None and args.expected_ranks < 1:
            raise ValueError("expected-ranks must be positive")
        if args.top < 1:
            raise ValueError("top must be positive")
        result = analyze(args.sqlite, args.expected_ranks, args.first_step, args.last_step, rank_pids, args.top)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if args.csv_dir:
        write_csv_tables(result, args.csv_dir)
    if args.strict_window and not result["capture_validation"]["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
