# SPDX-License-Identifier: GPL-3.0-or-later
#
# decode.py -- decode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections.abc import Callable
from dataclasses import dataclass

from fetch import VideoMetadata
from frame import (
    StreamConfig,
    pad_width,
    unpad_slice,
)


@dataclass
class DecoderConfig:
    stream_config: StreamConfig
    video_metadata: VideoMetadata


class Decoder:
    def __init__(self, config: DecoderConfig):
        self.config: DecoderConfig = config

        height: int = config.video_metadata.height
        width: int = config.video_metadata.width
        block_size: int = config.stream_config.block_size

        self._unpad_frame: Callable[[np.ndarray], np.ndarray] = self.unpad_frame(
            pad_width(height, width, block_size)
        )

    @staticmethod
    def unpad_frame(
        pad_width: tuple[tuple[int, int], tuple[int, int]]
    ) -> Callable[[np.ndarray], np.ndarray]:
        precomputed_unpad_slice = unpad_slice(pad_width)

        def unpad_frame(x: np.ndarray) -> np.ndarray:
            unpad_slice = precomputed_unpad_slice

            return x[..., *unpad_slice]

        return unpad_frame
