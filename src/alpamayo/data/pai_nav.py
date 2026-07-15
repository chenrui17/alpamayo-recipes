# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PAI dataset with navigation / text annotations loaded from a JSON file.

Example JSON (``nav_demo_samples.json``)::

    [
      {
        "clip_id": "1ae17e2a-...",
        "t0": 29708785000,
        "nav_text": "Turn left in 11m",
        "cot": "Yield to the cross-traffic vehicle ..."
      },
      ...
    ]
"""

import json
import logging
from typing import Any

from alpamayo.data.pai import PAIDataset
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

logger = logging.getLogger(__name__)


class PAIDatasetWithNav(PAIDataset):
    """PAIDataset driven by a JSON annotation file.

    Each entry in the JSON list becomes a separate sample. The ``t0`` field
    (microseconds) is used as the sampling timestamp, and text fields
    (``nav_text``, ``cot``, etc.) are merged into the sample dict.
    """

    def __init__(self, annotations_path: str, **kwargs: Any):
        """Initialize the dataset.

        Args:
            annotations_path: Path to a JSON file containing a list of sample
                dicts.  Each dict must have ``clip_id`` and ``t0``.
            **kwargs: Forwarded to :class:`PAIDataset` (``local_dir``,
                ``model_config``, ``vla_preprocess_args``, etc.).
        """
        super().__init__(**kwargs)

        with open(annotations_path) as f:
            samples: list[dict[str, Any]] = json.load(f)

        # Honour `chunk_ids` the same way plain PAIDataset does: only iterate
        # samples whose clip lives in one of the configured chunks. Without
        # this filter, nav training would silently iterate samples whose
        # chunk files aren't on disk and crash mid-batch with FileNotFoundError.
        if self.avdi.chunk_ids is not None:
            allowed_clip_ids = set(self.avdi.get_all_clip_ids())
            kept = [s for s in samples if s["clip_id"] in allowed_clip_ids]
            dropped = len(samples) - len(kept)
            if not kept:
                sample_clip_ids = sorted({s["clip_id"] for s in samples})
                raise ValueError(
                    f"[PAIDatasetWithNav] All {len(samples)} annotated samples in "
                    f"{annotations_path} were filtered out by chunk_ids="
                    f"{self.avdi.chunk_ids}. None of the annotation clip_ids "
                    f"({sample_clip_ids[:3]}{'...' if len(sample_clip_ids) > 3 else ''}) "
                    f"are in those chunks. Either widen chunk_ids to cover the "
                    f"annotated clips' chunks, or use an annotations file whose "
                    f"clips live in the configured chunks."
                )
            if dropped:
                logger.warning(
                    "[PAIDatasetWithNav] Filtered out %d/%d annotated samples whose "
                    "clip chunks aren't in chunk_ids=%s; keeping %d.",
                    dropped, len(samples), self.avdi.chunk_ids, len(kept),
                )
            self._samples = kept
        else:
            self._samples = samples

        self.clip_ids = [s["clip_id"] for s in self._samples]
        import os as _os
        # Number of unique samples (cache is keyed modulo this).
        self._orig_len = len(self._samples)

        # OPT: for small multi-epoch finetune sets (stage2 nav), per-sample decode
        # (~1.6s) dominates and per-worker caches thrash under rank/worker index
        # sharding. Preload all fully-processed UNIQUE samples into memory here in the
        # main process; DataLoader workers fork after this and inherit the cache copy-
        # on-write, making __getitem__ a pure memory return (near-zero data_wait).
        if _os.environ.get("APPLY_PRELOAD", "0") == "1":
            self._sample_cache = {}
            _prev = _os.environ.get("APPLY_SAMPLE_CACHE")
            _os.environ["APPLY_SAMPLE_CACHE"] = "1"
            for _i in range(self._orig_len):
                self[_i]
            if _prev is None:
                _os.environ.pop("APPLY_SAMPLE_CACHE", None)
            else:
                _os.environ["APPLY_SAMPLE_CACHE"] = _prev
            logger.info("[PAIDatasetWithNav] APPLY_PRELOAD: cached %d unique samples in memory", len(self._sample_cache))

        # BENCH: optionally repeat the sample list to lengthen epochs so tiny-dataset
        # epoch-boundary reshuffle stalls do not dominate throughput measurement. Done
        # AFTER preload; repeated indices reuse the cache via modulo in __getitem__.
        _rep = int(_os.environ.get("APPLY_REPEAT_DATA", "1"))
        if _rep > 1:
            self._samples = self._samples * _rep
            self.clip_ids = self.clip_ids * _rep

    def __len__(self) -> int:
        """Return the number of annotated samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Load a sample using the annotation's clip_id and t0."""
        # OPT: optional in-memory cache of fully-processed samples. Stage2 nav has a
        # small sample set trained for multiple epochs, so decoding+tokenizing every
        # epoch is pure repeated work. Gated by APPLY_SAMPLE_CACHE (per-worker cache).
        import os as _os
        _ckey = idx % getattr(self, "_orig_len", len(self._samples))
        if _os.environ.get("APPLY_SAMPLE_CACHE", "0") == "1":
            if not hasattr(self, "_sample_cache"):
                self._sample_cache = {}
            if _ckey in self._sample_cache:
                return self._sample_cache[_ckey]
        entry = self._samples[idx]
        clip_id = entry["clip_id"]
        t0_us = int(entry["t0_relative"])

        _prof_load = _os.environ.get("APPLY_LOAD_PROFILE", "0") == "1"
        if _prof_load:
            import time as _tl
            _lt0 = _tl.perf_counter()

        sample_data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        sample_data["nav_text"] = entry["nav_text"]

        for key in sample_data.keys():
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        if _prof_load:
            import time as _tl2
            print(f"[LOADTIME] idx={idx} total={( _tl2.perf_counter()-_lt0)*1000:.1f}ms", flush=True)

        # OPT: after preprocessing, raw pixel tensors are dead weight for the training
        # forward (it only consumes tokenized_data/pixel_values); dropping them avoids
        # multi-MB worker->main IPC + pin + H2D copies every step. Gated, and only when
        # not in generation/eval mode (eval may still want raw frames).
        if _os.environ.get("APPLY_DROP_RAW", "0") == "1" and not sample_data.get("generation_mode", False):
            sample_data.pop("image_frames", None)

        if _os.environ.get("APPLY_SAMPLE_CACHE", "0") == "1":
            self._sample_cache[_ckey] = sample_data

        return sample_data
