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
class EncoderConfig:
    stream_config: StreamConfig
    video_metadata: VideoMetadata


class Encoder:
    def __init__(self, config: EncoderConfig):
        self.config: EncoderConfig = config

        height: int = config.video_metadata.height
        width: int = config.video_metadata.width
        block_size: int = config.stream_config.block_size

        self._pad_frame: Callable[[np.ndarray], np.ndarray] = self.pad_frame(
            height, width, block_size
        )

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
