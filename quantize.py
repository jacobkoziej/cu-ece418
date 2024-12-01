# SPDX-License-Identifier: GPL-3.0-or-later
#
# quantize.py -- quantize frames
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from abc import (
    ABC,
    abstractmethod,
)


class Quantizer(ABC):
    @abstractmethod
    def quantize(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dequantize(self, x: np.ndarray) -> np.ndarray:
        pass
