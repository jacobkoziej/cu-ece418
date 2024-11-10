# SPDX-License-Identifier: GPL-3.0-or-later
#
# encode.py -- encode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    block_size: int = 16
    height: int
    width: int
