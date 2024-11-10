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
