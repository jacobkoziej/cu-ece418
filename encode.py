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
        assert target.ndim == 2
        assert 2 <= search.ndim <= 3

        assert target.shape == search.shape[-2:]
        assert target.dtype == search.dtype

        height, width = target.shape
        block_size = self.config.stream.block_size
        search_limit = self.config.search_limit

        assert not height % block_size
        assert not width % block_size

        blocked_height = height // block_size
        blocked_width = width // block_size

        search_range = block_size * search_limit

        motion_vectors = np.zeros(
            (blocked_height, blocked_width, search.ndim), dtype=np.int32
        )
        residuals = np.zeros(
            (blocked_height, blocked_width, block_size, block_size), dtype=target.dtype
        )

        if search.ndim == 3:
            target = np.expand_dims(target, axis=0)

        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                block = target[..., i : i + block_size, j : j + block_size]

                min_sad = float("inf")
                best_match = (0, 0)

                for y in range(
                    max(0, i - search_range),
                    min(height - block_size + 1, i + search_range + 1),
                    block_size,
                ):
                    for x in range(
                        max(0, j - search_range),
                        min(width - block_size + 1, j + search_range + 1),
                        block_size,
                    ):
                        candidate = search[..., y : y + block_size, x : x + block_size]

                        sad = np.sum(np.abs(block - candidate), axis=(-2, -1))

                        if sad.size == 2:
                            forward_prediction = np.argmin(sad)
                            sad = sad[forward_prediction]

                        if sad < min_sad:
                            min_sad = sad

                            best_match = ((x - j) // block_size, (y - i) // block_size)

                            if candidate.ndim == 3:
                                best_match = (forward_prediction,) + best_match
                                candidate = candidate[forward_prediction]

                            residual = block - candidate

                block_i = i // block_size
                block_j = j // block_size

                motion_vectors[block_i, block_j] = best_match
                residuals[block_i, block_j] = residual.squeeze()

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
