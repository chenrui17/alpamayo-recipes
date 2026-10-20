"""Tiny SQLite fixtures verify Nsight timeline accounting without Nsight/GPU."""

from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .analyze_nsys import Export, analyze, api_category, kernel_group, merge, subtract, write_csv_tables


def fixture(path: Path, ranks: int = 2) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER, eventType INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
            globalPid INTEGER, correlationId INTEGER, demangledName INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER);
        CREATE TABLE OSRT_API(start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER);
    """)
    strings = {
        1: "void ampere_gemm()", 2: "flash_fwd_kernel", 3: "ncclDevKernel_AllReduce",
        4: "at::vectorized_elementwise_kernel", 5: "cudaLaunchKernel", 6: "cudaDeviceSynchronize",
        7: "cuCtxSynchronize", 8: "poll", 9: "vlm", 10: "visual", 11: "optimizer_step_5",
        12: "optimizer_step_7", 13: "language", 14: "head", 15: "expert", 16: "action",
        17: "trajectory", 18: "next_batch", 19: "pthread_cond_wait",
    }
    connection.executemany("INSERT INTO StringIds VALUES (?, ?)", strings.items())
    for device in range(ranks):
        pid = 101 + device
        tid = (pid << 24) | 7
        for step, start in zip(range(5, 9), (0, 120, 240, 360)):
            # Mix literal text with registered string IDs, as real exports do.
            text_id = {5: 11, 7: 12}.get(step)
            text = None if text_id else f"optimizer_step_{step}"
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, 59)",
                               (start, start + 100, tid, text, text_id))
        for start, end, name_id in ((10, 90, 9), (20, 40, 10), (40, 80, 13), (130, 210, 14),
                                    (250, 330, 15), (365, 400, 16), (400, 450, 17), (100, 120, 18)):
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL, ?, 59)", (start, end, tid, name_id))
        # Correlation IDs intentionally repeat across ranks. Kernel grouping uses
        # actual GPU durations; two streams overlap during step 5.
        for start, end, stream, correlation, name_id, launch in (
            (25, 55, 1, 100, 1, 22), (35, 65, 2, 101, 2, 32), (80, 100, 3, 102, 3, 81),
            (135, 155, 1, 103, 4, 131), (260, 300, 1, 104, 1, 252), (380, 420, 1, 105, 4, 370),
        ):
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (start, end, device, stream, pid << 24, correlation, name_id, name_id))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, 5)",
                               (launch, launch + 2, tid, correlation))
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (90, 100, ?, 200, 6)", (tid,))
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (92, 99, ?, 201, 7)", (tid,))
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (170, 180, ?, ?)", (device, pid << 24))
        connection.execute("INSERT INTO OSRT_API VALUES (105, 115, ?, 8)", (tid,))
        connection.execute("INSERT INTO OSRT_API VALUES (12, 22, ?, 19)", (tid,))
    connection.commit()
    connection.close()


class NsightAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "tiny.sqlite"
        fixture(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_kernel_duration_and_device_union_do_not_add_overlaps(self):
        result = analyze(self.path, expected_ranks=2, rank_pids={101: 0, 102: 1})
        self.assertTrue(result["capture_validation"]["valid"])
        self.assertEqual(result["capture_validation"]["observed_rank_processes"], 2)
        self.assertEqual(result["capture_validation"]["selected_window_union_ns"], 460)
        self.assertEqual(len(result["optimizer_steps"]), 8)
        self.assertTrue(all(row["overlapping_gpu_kernel_count"] > 0 for row in result["optimizer_steps"]))
        for device in result["devices"]:
            self.assertEqual(device["window_union_ns"], 460)
            self.assertEqual(device["kernel_busy_union_ns"], 160)
            self.assertEqual(device["gpu_activity_busy_union_ns"], 170)
            self.assertEqual(device["no_recorded_gpu_activity_ns"], 290)
            groups = [row for row in result["kernels_by_device_group"] if row["device"] == device["device"]]
            self.assertEqual(sum(row["sum_kernel_duration_ns"] for row in groups), 180)
            self.assertEqual({row["group"] for row in groups}, {"NCCL", "FA", "GEMM", "pointwise"})
        self.assertTrue(any("eight-GPU sum" in entry for entry in result["interpretation"]))

    def test_nvtx_registered_string_resolution_and_nested_exclusive_time(self):
        result = analyze(self.path, expected_ranks=2)
        phases = {row["phase"]: row for row in result["nvtx_phases"] if row["pid"] == 101}
        self.assertEqual(phases["vlm"]["inclusive_thread_union_ns"], 80)
        self.assertEqual(phases["vlm"]["exclusive_recognized_phase_ns"], 20)
        self.assertEqual(phases["visual"]["exclusive_recognized_phase_ns"], 20)
        self.assertEqual(phases["language"]["exclusive_recognized_phase_ns"], 40)
        self.assertEqual(phases["vlm"]["example_names"], ["vlm"])
        self.assertEqual({row["step"] for row in result["optimizer_steps"]}, {5, 6, 7, 8})
        # A kernel launched in the action phase extends beyond its host NVTX end;
        # correlation-based attribution still assigns its entire GPU duration.
        action = next(row for row in result["kernels_by_device_phase"] if row["device"] == "0" and row["phase"] == "action")
        self.assertEqual(action["sum_kernel_duration_ns"], 40)
        export = Export(self.path)
        try:
            self.assertEqual(export.name({"text": "123", "textid": 9}, nvtx=True), "123")
            self.assertEqual(export.name({"text": None, "textid": 9}, nvtx=True), "vlm")
        finally:
            export.connection.close()

    def test_cuda_runtime_driver_nested_calls_union_and_wait_clues(self):
        result = analyze(self.path, expected_ranks=2)
        sync_calls = [row for row in result["cuda_api"] if row["pid"] == 101 and row["category"] == "explicit_sync"]
        self.assertEqual(sum(row["sum_host_api_duration_ns"] for row in sync_calls), 17)
        sync_union = next(row for row in result["cuda_api_category_unions"] if row["pid"] == 101 and row["category"] == "explicit_sync")
        self.assertEqual(sync_union["host_api_union_across_tables_threads_ns"], 10)
        poll = next(row for row in result["cpu_data_wait_clues"]["osrt_wait_calls"] if row["pid"] == 101 and row["name"] == "poll")
        self.assertEqual(poll["host_wait_union_ns"], 10)
        self.assertEqual(poll["overlap_with_process_gpu_idle_union_ns"], 10)
        self.assertEqual(len(result["cpu_data_wait_clues"]["interstep_gaps"]), 6)
        self.assertEqual(len(result["cpu_data_wait_clues"]["nvtx_data_ranges"]), 2)
        self.assertEqual(api_category("cudaStreamWaitEvent"), "device_wait_enqueue")
        self.assertEqual(api_category("cudaMemcpyAsync"), "async_memcpy_enqueue")

    def test_missing_rank_or_incomplete_step_is_rejected(self):
        self.assertFalse(analyze(self.path, expected_ranks=8)["capture_validation"]["valid"])
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE NVTX_EVENTS SET end=NULL WHERE text='optimizer_step_8' AND globalTid=?", ((102 << 24) | 7,))
        connection.commit()
        connection.close()
        result = analyze(self.path, expected_ranks=2)
        self.assertFalse(result["capture_validation"]["valid"])
        process = next(row for row in result["capture_validation"]["processes"] if row["pid"] == 102)
        self.assertEqual(process["missing_steps"], [8])
        self.assertEqual(process["open_or_invalid_steps"], [8])

    def test_absent_tables_and_legacy_kernel_schema(self):
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            DROP TABLE NVTX_EVENTS;
            DROP TABLE OSRT_API;
            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME TO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
        """)
        connection.close()
        result = analyze(self.path, expected_ranks=2)
        self.assertFalse(result["capture_validation"]["valid"])
        self.assertTrue(result["kernels_by_device_group"])
        self.assertTrue(any("NVTX_EVENTS" in warning for warning in result["warnings"]))

    def test_last_step_label_without_gpu_events_is_rejected(self):
        connection = sqlite3.connect(self.path)
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=1 AND start >= 360")
        connection.commit()
        connection.close()
        result = analyze(self.path, expected_ranks=2)
        self.assertFalse(result["capture_validation"]["valid"])
        missing = result["capture_validation"]["steps_without_gpu_kernels"]
        self.assertEqual([(row["pid"], row["step"]) for row in missing], [(102, 8)])

    def test_export_csv_and_input_database_stays_unchanged(self):
        before = self.path.read_bytes()
        result = analyze(self.path, expected_ranks=2)
        directory = Path(self.temporary.name) / "csv"
        write_csv_tables(result, directory)
        self.assertEqual(before, self.path.read_bytes())
        with (directory / "kernels_by_device_group.csv").open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["device"] for row in rows}, {"0", "1"})

    def test_interval_helpers_and_conservative_kernel_classification(self):
        self.assertEqual(merge([(1, 4), (3, 7), (7, 9)]), [(1, 9)])
        self.assertEqual(subtract([(0, 10)], [(2, 4), (3, 8)]), [(0, 2), (8, 10)])
        self.assertEqual(kernel_group("layer_norm_cuda"), "other")
        self.assertEqual(kernel_group("flash::attention_gemm"), "FA")


if __name__ == "__main__":
    unittest.main()
