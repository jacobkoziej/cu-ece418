# SPDX-License-Identifier: GPL-3.0-or-later
#
# fetch.py -- fetch video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    frame_rate: float
    height: int
    width: int
