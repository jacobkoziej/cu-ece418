# SPDX-License-Identifier: GPL-3.0-or-later
#
# encode.py -- encode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

from dataclasses import dataclass

from fetch import VideoMetadata


@dataclass
class EncoderConfig:
    video_metadata: VideoMetadata
    block_size: int = 16


class Encoder:
    def __init__(self, config: EncoderConfig):
        self.config: EncoderConfig = config

        height: int = config.video_metadata.height
        width: int = config.video_metadata.width
        block_size: int = config.block_size

        self._pad_frame: Callable[[np.ndarray], np.ndarray] = self.pad_frame(
            height, width, block_size
        )

    @staticmethod
    def pad_frame(
        height: int,
        width: int,
        block_size: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        target_height: int = height + (height % block_size)
        target_width: int = width + (width % block_size)

        pad_top: int = (target_height - height) // 2
        pad_bottom: int = target_height - height - pad_top
        pad_left: int = (target_width - width) // 2
        pad_right: int = target_width - width - pad_left

        def pad_frame(x: np.ndarray) -> np.ndarray:
            pad_width: tuple[tuple[int, int], ...] = (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
            )

            if x.ndim > 2:
                pad_width = (((0, 0),) * (x.ndim - 2)) + pad_width

            return np.pad(x, pad_width, "edge")

        return pad_frame
