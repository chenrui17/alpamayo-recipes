"""CPU-only contract tests; neither torch nor transformers needs to be installed."""

from __future__ import annotations

import importlib.util
import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from .analyze import statistics, summarize
from .run import parse_args, verify_result


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def load_callback(clock, state, rank=0, world_size=1):
    events = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: events.append(("reset_peak", state.global_step)),
        max_memory_allocated=lambda: 1000 + state.global_step,
        max_memory_reserved=lambda: 2000 + state.global_step,
    )

    def synchronize():
        events.append(("sync", state.global_step))
        clock.now += 0.02

    def profiler_event(name):
        events.append((name, state.global_step))
        return 0

    cuda.synchronize = synchronize
    cuda.nvtx = types.SimpleNamespace(
        range_push=lambda name: events.append(("nvtx_push", name)),
        range_pop=lambda: events.append(("nvtx_pop", state.global_step)),
    )
    cuda.cudart = lambda: types.SimpleNamespace(
        cudaProfilerStart=lambda: profiler_event("cudaProfilerStart"),
        cudaProfilerStop=lambda: profiler_event("cudaProfilerStop"),
    )

    def gather(outputs, payload):
        events.append(("gather", rank))
        for index in range(world_size):
            item = json.loads(json.dumps(payload))
            item["rank"] = index
            if index != rank:
                for record in item["records"]:
                    if record["step_seconds"] is not None:
                        record["step_seconds"] += 0.5
                item["max_memory_allocated_bytes"] += 100
            outputs[index] = item

    distributed = types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: world_size > 1,
        get_rank=lambda: rank, get_world_size=lambda: world_size,
        all_gather_object=gather,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = cuda
    fake_torch.distributed = distributed
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.TrainerCallback = object
    spec = importlib.util.spec_from_file_location("fake_step_callback", Path(__file__).with_name("callback.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
        spec.loader.exec_module(module)
    return module, events


class CallbackTests(unittest.TestCase):
    def run_callback(self, directory, profile=False, max_steps=20, final_step=None, world_size=2, rank=0):
        state = types.SimpleNamespace(global_step=0)
        clock = Clock()
        module, events = load_callback(clock, state, rank=rank, world_size=world_size)
        output = Path(directory) / "steps.json"
        callback = module.DistributedStepCallback(
            str(output), "test-run", steady_end_step=max_steps, expected_max_steps=max_steps,
            profile=profile, global_samples_per_step=8,
        )
        with patch.object(module.time, "perf_counter", clock):
            callback.on_train_begin(None, state, None)
            for step in range(1, (final_step or max_steps) + 1):
                callback.on_step_begin(None, state, None)
                clock.now += 1.0
                state.global_step = step
                callback.on_step_end(None, state, None)
                callback.on_log(None, state, None, logs={"loss": 0.5 / step, "other": object()})
            callback.on_train_end(None, state, None)
        return output, events

    def test_all_ranks_synchronize_and_rank_max_is_computed_per_step(self):
        with tempfile.TemporaryDirectory() as directory:
            rng_before = random.getstate()
            output, events = self.run_callback(directory)
            self.assertEqual(random.getstate(), rng_before)
            result = json.loads(output.read_text())
        self.assertTrue(result["complete"])
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(len([event for event in events if event[0] == "sync"]), 20)
        self.assertIn(("gather", 0), events)
        self.assertFalse(any(event[0].startswith("cudaProfiler") or event[0].startswith("nvtx") for event in events))
        self.assertIsNone(result["ranks"][0]["records"][0]["step_seconds"])
        self.assertAlmostEqual(result["ranks"][0]["records"][1]["step_seconds"], 1.02)
        self.assertAlmostEqual(result["rank_max_records"][4]["step_seconds"], 1.52)
        self.assertEqual(result["rank_max_records"][4]["slowest_rank"], 1)
        self.assertEqual(len(result["ranks"][0]["logs"]), 20)
        self.assertNotIn("other", result["ranks"][0]["logs"][0])
        summary = summarize(result)
        self.assertEqual(summary["rank_max"]["count"], 16)
        self.assertAlmostEqual(summary["rank_max"]["mean_seconds"], 1.52)
        self.assertAlmostEqual(summary["rank_max"]["samples_per_second"], 8 / 1.52)
        self.assertAlmostEqual(summary["per_rank"][0]["mean_sync_seconds"], 0.02)
        self.assertEqual(summary["max_memory_allocated_bytes"], 1120)

    def test_nonzero_rank_also_gathers_but_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_callback(directory, rank=1)
            self.assertFalse(output.exists())
            self.assertIn(("gather", 1), events)

    def test_profile_starts_before_step5_and_stops_after_step8(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_callback(directory, profile=True)
            result = json.loads(output.read_text())
        self.assertTrue(result["complete"])
        self.assertEqual([event for event in events if event[0] == "cudaProfilerStart"], [("cudaProfilerStart", 4)])
        self.assertEqual([event for event in events if event[0] == "cudaProfilerStop"], [("cudaProfilerStop", 8)])
        self.assertLess(events.index(("cudaProfilerStart", 4)), events.index(("nvtx_push", "optimizer_step_5")))
        self.assertLess(events.index(("sync", 8)), events.index(("cudaProfilerStop", 8)))
        self.assertEqual(len([event for event in events if event[0] == "nvtx_push"]), 20)
        self.assertEqual(result["ranks"][0]["capture_started_at_step"], 5)
        self.assertEqual(result["ranks"][0]["capture_stopped_at_step"], 8)

    def test_early_stop_is_incomplete_even_if_profiler_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_callback(directory, profile=True, final_step=6)
            result = json.loads(output.read_text())
        self.assertFalse(result["complete"])
        self.assertIn(("cudaProfilerStop", 6), events)
        self.assertIsNone(result["ranks"][0]["capture_stopped_at_step"])
        with self.assertRaisesRegex(ValueError, "did not complete"):
            summarize(result)

    def test_callback_does_not_need_cuda_for_cpu_timings(self):
        state = types.SimpleNamespace(global_step=0)
        clock = Clock()
        module, _ = load_callback(clock, state)
        module.torch.cuda.is_available = lambda: False
        with tempfile.TemporaryDirectory() as directory:
            callback = module.DistributedStepCallback(str(Path(directory) / "result.json"), "cpu")
            with patch.object(module.time, "perf_counter", clock):
                callback.on_train_begin(None, state, None)
                for step in range(1, 21):
                    clock.now += 1
                    state.global_step = step
                    callback.on_step_end(None, state, None)
                callback.on_train_end(None, state, None)
            result = json.loads((Path(directory) / "result.json").read_text())
        self.assertIsNone(result["ranks"][0]["max_memory_allocated_bytes"])
        self.assertTrue(result["complete"])


class AnalysisTests(unittest.TestCase):
    def test_statistics_p90_definition(self):
        result = statistics([1, 2, 3, 4, 5])
        self.assertEqual(result["mean_seconds"], 3)
        self.assertEqual(result["median_seconds"], 3)
        self.assertAlmostEqual(result["p90_seconds"], 4.6)
        for invalid in ([], [0], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                statistics(invalid)

    def test_result_verification_rejects_stale_identity_and_rank_count(self):
        args = types.SimpleNamespace(mode="timing", nproc_per_node=2, max_steps=20,
                                     steady_start_step=5, steady_end_step=20)
        with self.assertRaisesRegex(RuntimeError, "run_id"):
            verify_result({"run_id": "old-run"}, args, "new-run")
        with self.assertRaisesRegex(RuntimeError, "world size"):
            verify_result({"run_id": "new-run", "mode": "timing", "world_size": 1}, args, "new-run")

    def test_cli_requires_new_run_name_and_protects_measurement_overrides(self):
        base = ["--work-dir", ".", "--result-dir", ".", "--config-path", "pkg://example", "--config-name", "stage2"]
        with self.assertRaises(SystemExit):
            parse_args([*base, "--run-id", "../old"])
        with self.assertRaises(SystemExit):
            parse_args([*base, "--override", "++trainer.max_steps=1"])
        args = parse_args([*base, "--override", "performance.zip_cache=false"])
        self.assertEqual(args.override, ["performance.zip_cache=false"])


if __name__ == "__main__":
    unittest.main()
