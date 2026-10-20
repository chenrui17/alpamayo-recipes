# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the opt-in training dataset wrapper and cache isolation."""

import os
import unittest
from unittest.mock import patch

from alpamayo1_5_sft.performance.stage2_data import (
    Stage2TrainingDataset,
    apply_stage2_training_data,
)


class RecordingDataset:
    def __init__(self, samples):
        self.samples = samples
        self.reads = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.samples[index]


class Stage2DataSwitchTests(unittest.TestCase):
    def test_disabled_returns_source_without_reading_samples(self):
        source = RecordingDataset([{"value": 1}])
        with patch.dict(os.environ, {}, clear=True):
            wrapped, report = apply_stage2_training_data(source)
        self.assertIs(wrapped, source)
        self.assertEqual(source.reads, [])
        self.assertEqual(report["cache_tensor_bytes"], 0)
        self.assertEqual(report["repeat_factor"], 1)

    def test_repeat_maps_original_indices_without_mutating_source(self):
        source = RecordingDataset([{"value": 1}, {"value": 2}])
        with patch.dict(os.environ, {"APPLY_REPEAT_DATA": "8"}, clear=True):
            wrapped, report = apply_stage2_training_data(source)
        self.assertEqual(len(wrapped), 16)
        self.assertEqual(len(source), 2)
        self.assertEqual([wrapped[i]["value"] for i in range(5)], [1, 2, 1, 2, 1])
        self.assertEqual(wrapped[-1]["value"], 2)
        self.assertEqual(source.reads, [0, 1, 0, 1, 0, 1])
        self.assertEqual(report["unique_samples"], 2)
        for index in (16, -17):
            with self.assertRaises(IndexError):
                wrapped[index]

    def test_drop_raw_preserves_source_and_online_processing_inputs(self):
        tokenized = {"image_frames": [1], "tokenized_data": {"text": "a"}}
        online = {"image_frames": [2]}
        unavailable = {"image_frames": [3], "tokenized_data": None}
        source = RecordingDataset([tokenized, online, unavailable])
        with patch.dict(os.environ, {"APPLY_DROP_RAW": "1"}, clear=True):
            wrapped, _ = apply_stage2_training_data(source)
        self.assertNotIn("image_frames", wrapped[0])
        self.assertIn("image_frames", tokenized)
        self.assertEqual(wrapped[1]["image_frames"], [2])
        self.assertEqual(wrapped[2]["image_frames"], [3])

    def test_evaluation_is_untouched_even_with_all_switches_enabled(self):
        source = RecordingDataset([{}])
        env = {"APPLY_PRELOAD": "1", "APPLY_REPEAT_DATA": "8", "APPLY_DROP_RAW": "1"}
        with patch.dict(os.environ, env, clear=True):
            wrapped, report = apply_stage2_training_data(source, training=False)
        self.assertIs(wrapped, source)
        self.assertEqual(source.reads, [])
        self.assertEqual(report["skipped_reason"], "evaluation_dataset")

    def test_bad_switch_and_repeat_fail_explicitly(self):
        with patch.dict(os.environ, {"APPLY_PRELOAD": "tru"}, clear=True):
            with self.assertRaisesRegex(ValueError, "APPLY_PRELOAD"):
                apply_stage2_training_data(RecordingDataset([{}]))
        with self.assertRaisesRegex(ValueError, "repeat_factor"):
            Stage2TrainingDataset(RecordingDataset([{}]), repeat_factor=0)

    def test_empty_repeated_dataset_has_no_valid_indices(self):
        wrapped = Stage2TrainingDataset(RecordingDataset([]), repeat_factor=8)
        self.assertEqual(len(wrapped), 0)
        with self.assertRaises(IndexError):
            wrapped[0]

    def test_repeat_argument_overrides_environment(self):
        source = RecordingDataset([{}])
        with patch.dict(os.environ, {"APPLY_REPEAT_DATA": "8"}, clear=True):
            wrapped, report = apply_stage2_training_data(source, repeat_factor=3)
        self.assertEqual(len(wrapped), 3)
        self.assertEqual(report["repeat_factor"], 3)


class Stage2PreloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"CPU torch/numpy unavailable: {exc}") from exc
        cls.torch = torch
        cls.numpy = numpy

    def test_hydra_sequence_metadata_is_cached_and_isolated(self):
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise unittest.SkipTest("OmegaConf unavailable") from exc
        original = OmegaConf.create(["traj_future", "reasoning"])
        source = RecordingDataset([{"label_components": original}])
        wrapped = Stage2TrainingDataset(source, preload=True)
        first = wrapped[0]
        self.assertEqual(first["label_components"], ["traj_future", "reasoning"])
        first["label_components"].append("mutation")
        self.assertEqual(wrapped[0]["label_components"], ["traj_future", "reasoning"])
        self.assertEqual(list(original), ["traj_future", "reasoning"])

    def test_preload_reads_unique_samples_once_and_isolates_every_return(self):
        torch, numpy = self.torch, self.numpy
        sample = {
            "tokenized_data": {"values": torch.tensor([1.0, 2.0])},
            "metadata": [{"array": numpy.array([3, 4])}],
            "image_frames": torch.ones(100),
        }
        source = RecordingDataset([sample])
        env = {"APPLY_PRELOAD": "1", "APPLY_REPEAT_DATA": "8", "APPLY_DROP_RAW": "1"}
        with patch.dict(os.environ, env, clear=True):
            wrapped, report = apply_stage2_training_data(source)
        self.assertEqual(source.reads, [0])
        self.assertEqual(len(wrapped), 8)
        self.assertEqual(report["cache_tensor_bytes"], 8)
        self.assertGreater(report["cache_estimated_bytes"], 8)
        self.assertGreaterEqual(report["preprocessing_seconds"], 0)
        first = wrapped[0]
        first["tokenized_data"]["values"].zero_()
        first["tokenized_data"].pop("values")
        first["metadata"][0]["array"][0] = 0
        sample["tokenized_data"]["values"].fill_(9)
        second = wrapped[1]
        torch.testing.assert_close(second["tokenized_data"]["values"], torch.tensor([1.0, 2.0]))
        numpy.testing.assert_array_equal(second["metadata"][0]["array"], [3, 4])
        self.assertEqual(second["tokenized_data"]["values"].device.type, "cpu")
        self.assertIn("image_frames", sample)
        self.assertNotIn("image_frames", second)
        self.assertEqual(source.reads, [0])

    def test_cache_budget_counts_arrays_and_stops_before_next_clone(self):
        source = RecordingDataset([{"array": self.numpy.ones(100, dtype="uint8")}])
        with self.assertRaisesRegex(MemoryError, "preload_max_bytes=50"):
            Stage2TrainingDataset(source, preload=True, preload_max_bytes=50)
        self.assertEqual(source.reads, [0])

    def test_preload_propagates_dataset_errors(self):
        class BrokenDataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index == 1:
                    raise OSError("missing camera zip")
                return {"value": 1}

        with self.assertRaisesRegex(OSError, "missing camera zip"):
            Stage2TrainingDataset(BrokenDataset(), preload=True)

    def test_preload_in_loader_worker_is_rejected(self):
        with patch.object(self.torch.utils.data, "get_worker_info", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "before DataLoader workers"):
                Stage2TrainingDataset(RecordingDataset([{}]), preload=True)

    def test_explicit_budget_overrides_environment_and_invalid_budget_fails(self):
        source = RecordingDataset([{"value": 1}])
        env = {"APPLY_PRELOAD": "1", "APPLY_PRELOAD_MAX_BYTES": "1"}
        with patch.dict(os.environ, env, clear=True):
            _, report = apply_stage2_training_data(source, preload_max_bytes=1024)
            self.assertEqual(report["preload_max_bytes"], 1024)
            with self.assertRaisesRegex(MemoryError, "preload_max_bytes=1"):
                apply_stage2_training_data(source)
        with self.assertRaisesRegex(ValueError, "preload_max_bytes"):
            Stage2TrainingDataset(source, preload=True, preload_max_bytes=0)


if __name__ == "__main__":
    unittest.main()
