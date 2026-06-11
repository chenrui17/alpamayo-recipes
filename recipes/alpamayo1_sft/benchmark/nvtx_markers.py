# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVTX range helpers for CPU and GPU threads (nsys-visible)."""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


def _load_nvtx_lib() -> ctypes.CDLL | None:
    candidates = [
        "libnvToolsExt.so.1",
    ]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    for base in sys.path:
        bundled = Path(base) / "nvidia" / "nvtx" / "lib" / "libnvToolsExt.so.1"
        if bundled.is_file():
            return ctypes.CDLL(str(bundled))
    return None


_NVTX = _load_nvtx_lib()
if _NVTX is not None:
    _NVTX.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    _NVTX.nvtxRangePop.argtypes = []


def local_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _push(name: str) -> None:
    if _NVTX is not None:
        _NVTX.nvtxRangePushA(name.encode("utf-8"))
    elif torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _pop() -> None:
    if _NVTX is not None:
        _NVTX.nvtxRangePop()
    elif torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    _push(name)
    try:
        yield
    finally:
        _pop()
