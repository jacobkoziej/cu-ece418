# SPDX-License-Identifier: GPL-3.0-or-later
#
# frame.py -- frame formats
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from dataclasses import dataclass


@dataclass
class BFrame:
    motion_vectors: np.ndarray
    residuals: int


@dataclass
class IFrame:
    frame: np.ndarray


@dataclass
class PFrame:
    motion_vectors: np.ndarray
    reference_frame: int
    residuals: int


def pad_width(
    height: int,
    width: int,
    block_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    target_height: int = height + (height % block_size)
    target_width: int = width + (width % block_size)

    pad_top: int = (target_height - height) // 2
    pad_bottom: int = target_height - height - pad_top
    pad_left: int = (target_width - width) // 2
    pad_right: int = target_width - width - pad_left

    pad_width: tuple[tuple[int, int], tuple[int, int]] = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
    )

    return pad_width
