# SPDX-License-Identifier: GPL-3.0-or-later
#
# encode.py -- encode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections.abc import Callable
from dataclasses import dataclass

from fetch import VideoMetadata
from frame import (
    StreamConfig,
    pad_width,
)


@dataclass
class EncoderFrameRateConfig:
    i: int = 32
    p: int = 8


@dataclass
class EncoderConfig:
    frame_rate: EncoderFrameRateConfig
    metadata: VideoMetadata
    stream: StreamConfig
    search_limit: int = 1


class Encoder:
    def __init__(self, config: EncoderConfig):
        self.config: EncoderConfig = config

        height: int = config.metadata.height
        width: int = config.metadata.width
        block_size: int = config.stream.block_size

        self._pad_frame: Callable[[np.ndarray], np.ndarray] = self.pad_frame(
            height, width, block_size
        )

    def block_match(
        self, search: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert search.ndim == 3
        assert target.ndim == 3

        assert target.shape[-2:] == search.shape[-2:]
        assert target.dtype == search.dtype

        height: int
        width: int
        _, height, width = target.shape

        block_size: int = self.config.stream.block_size

        assert not height % block_size
        assert not width % block_size

        blocked_height: int = height // block_size
        blocked_width: int = width // block_size

        search_limit: int = self.config.search_limit
        search_range: int = block_size * search_limit

        motion_vectors: np.ndarray = np.zeros(
            shape=(blocked_height, blocked_width, 3),
            dtype=np.int32,
        )
        residuals: np.ndarray = np.zeros(
            shape=(blocked_height, blocked_width) + (block_size,) * 2,
            dtype=target.dtype,
        )

        def range_process(limit: int) -> range:
            return range(0, limit - block_size + 1, block_size)

        def range_search(position: int, limit: int) -> range:
            return range(
                max(0, position - search_range),
                min(limit - block_size + 1, position + search_range + 1),
                block_size,
            )

        def select_block(x: int, y: int) -> tuple[slice, slice]:
            block = (
                slice(x, x + block_size),
                slice(y, y + block_size),
            )

            return block

        def pel2block(x: int) -> int:
            return x // block_size

        def select_output_block(x: int, y: int) -> tuple[slice, slice]:
            output_block = (
                slice(x, x + block_size),
                slice(y, y + block_size),
            )

            return output_block

        def process_candidate(
            i: int,
            j: int,
            x: int,
            y: int,
            threshold: int,
        ) -> tuple[int, np.ndarray, np.ndarray]:
            candidate: np.ndarray = search[..., *select_block(y, x)]

            residual: np.ndarray = block - candidate

            sad: np.ndarray = np.sum(np.abs(residual), axis=(-2, -1))

            forward_prediction: int = int(np.argmin(sad))

            sad: int = sad[forward_prediction]

            residual = residual[forward_prediction]

            best_match: np.ndarray = np.array(
                [forward_prediction, pel2block(y - i), pel2block(x - j)]
            )

            return (sad, best_match, residual)

        def search_area(i, j, block) -> tuple[np.ndarray, np.ndarray]:
            sad: float = float("inf")

            for y in range_search(i, height):
                for x in range_search(j, width):
                    s, b, r = process_candidate(i, j, x, y, sad)

                    if s < sad:
                        sad = s
                        best_match = b
                        residual = r

            return (best_match, residual)

        for i in range_process(height):
            for j in range_process(width):
                block: np.ndarray = target[..., *select_block(i, j)]

                best_match, residual = search_area(i, j, block)

                block_i = pel2block(i)
                block_j = pel2block(j)

                motion_vectors[block_i, block_j] = best_match
                residuals[block_i, block_j] = residual

        return (motion_vectors, residuals)

    @staticmethod
    def pad_frame(
        height: int,
        width: int,
        block_size: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        precomputed_pad_width = pad_width(height, width, block_size)

        def pad_frame(x: np.ndarray) -> np.ndarray:
            pad_width = precomputed_pad_width

            if x.ndim > 2:
                pad_width = (((0, 0),) * (x.ndim - 2)) + pad_width

            return np.pad(x, pad_width, "edge")

        return pad_frame
