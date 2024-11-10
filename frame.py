# SPDX-License-Identifier: GPL-3.0-or-later
#
# frame.py -- frame formats
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from dataclasses import dataclass


@dataclass
class IFrame:
    frame: np.ndarray
