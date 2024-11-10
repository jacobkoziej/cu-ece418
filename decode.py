# SPDX-License-Identifier: GPL-3.0-or-later
#
# decode.py -- decode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    stream_config: StreamConfig
    video_metadata: VideoMetadata


class Decoder:
    def __init__(self, config: DecoderConfig):
        self.config: DecoderConfig = config
