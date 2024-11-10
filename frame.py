# SPDX-License-Identifier: GPL-3.0-or-later
#
# frame.py -- frame formats
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from dataclasses import dataclass


@dataclass
class BFrame:
    motion_vectors: np.ndarray
    residuals: np.ndarray


@dataclass
class IFrame:
    frame: np.ndarray


@dataclass
class PFrame:
    motion_vectors: np.ndarray
    reference_frame: int
    residuals: np.ndarray


@dataclass
class StreamConfig:
    block_size: int = 16


def pad_width(
    height: int,
    width: int,
    block_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    def pad(x: int, size: int) -> int:
        return x + ((size - (x % size)) % size)

    target_height: int = pad(height, block_size)
    target_width: int = pad(width, block_size)

    pad_top: int = (target_height - height) // 2
    pad_bottom: int = target_height - height - pad_top
    pad_left: int = (target_width - width) // 2
    pad_right: int = target_width - width - pad_left

    pad_width: tuple[tuple[int, int], tuple[int, int]] = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
    )

    return pad_width


def unpad_slice(
    pad_width: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[slice, slice]:
    height_pad, width_pad = pad_width

    def zero_to_none(x: int) -> int | None:
        return x if x else None

    height_slice = slice(
        zero_to_none(height_pad[0]),
        zero_to_none(-1 * height_pad[1]),
    )
    width_slice = slice(
        zero_to_none(width_pad[0]),
        zero_to_none(-1 * width_pad[1]),
    )

    unpad_slice = (height_slice, width_slice)

    return unpad_slice
