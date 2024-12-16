# SPDX-License-Identifier: GPL-3.0-or-later
#
# quantize.py -- quantize frames
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import Final

from einops import rearrange
from scipy.fft import (
    dctn,
    idctn,
)


@dataclass
class QuantizedValues:
    dc: np.ndarray
    ac: np.ndarray
    indices: np.ndarray


class Quantizer(ABC):
    @abstractmethod
    def quantize(self, x: np.ndarray) -> QuantizedValues:
        pass

    @abstractmethod
    def dequantize(self, quantized_values: QuantizedValues) -> np.ndarray:
        pass


class Magnitude(Quantizer):
    _block_axes: Final[tuple[int, int]] = (-2, -1)

    def __init__(
        self,
        quality: int,
        *,
        block_shape: tuple[int, int] = (16, 16),
    ) -> None:
        assert len(block_shape) == 2
        assert block_shape[0] >= 1
        assert block_shape[1] >= 1

        flat_shape: int = block_shape[0] * block_shape[1]

        assert flat_shape <= 256
        assert flat_shape >= 1

        assert quality <= flat_shape
        assert quality >= 1

        self._quality: Final[int] = quality

        self._block_shape: Final[tuple[int, int]] = block_shape
        self._flat_shape: Final[int] = flat_shape

    def quantize(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert x.ndim >= 3

        block_shape: Final[tuple[int, int]] = self._block_shape

        assert x.shape[-2:] == block_shape

        base_shape: Final[tuple[int, ...]] = x.shape[:-2]

        x = rearrange(x, "... y x -> (...) y x")

        block_axes: Final[tuple[int, int]] = self._block_axes

        dc: np.ndarray
        dc = np.mean(x, axis=block_axes)
        dc = np.expand_dims(dc, block_axes)

        ac: np.ndarray
        ac = dctn(x - dc, axes=block_axes, norm="ortho", type=2)
        ac = rearrange(ac, "... y x -> ... (y x)")

        quality: Final[int] = self._quality

        indices: np.ndarray = np.abs(ac).argsort()[..., -quality:]

        ac = np.stack([a[..., i] for (i, a) in zip(indices, ac)])

        dc = dc.astype(np.float16)
        ac = ac.astype(np.float16)
        indices = indices.astype(np.uint8)

        dc = dc.reshape(*base_shape, -1)
        ac = ac.reshape(*base_shape, -1)
        indices = indices.reshape(*base_shape, -1)

        return QuantizedValues(
            dc=dc,
            ac=ac,
            indices=indices,
        )

    def dequantize(self, quantized_values: QuantizedValues) -> np.ndarray:
        dc: np.ndarray = quantized_values.dc
        ac: np.ndarray = quantized_values.ac
        indices: np.ndarray = quantized_values.indices

        assert dc.ndim >= 2
        assert ac.ndim >= 2
        assert indices.ndim >= 2

        assert dc.shape[:-1] == ac.shape[:-1]
        assert dc.shape[:-1] == indices.shape[:-1]

        input_shape: Final[tuple[int, ...]] = dc.shape[:-1]
        flat_shape: Final[int] = self._flat_shape

        x: np.ndarray
        x = np.zeros((*input_shape, flat_shape))
        x = rearrange(x, "... f -> (...) f")

        ac = rearrange(ac, "... f -> (...) f")
        indices = rearrange(indices, "... f -> (...) f")

        for i, ac_i, x_i in zip(indices, ac, x):
            x_i[..., i] = ac_i

        block_shape: Final[tuple[int, int]] = self._block_shape

        x = x.reshape(*input_shape, -1)
        x = rearrange(
            x,
            "... (y x) -> ... y x",
            y=block_shape[0],
            x=block_shape[1],
        )
        dc = np.expand_dims(dc, -1)

        block_axes: Final[tuple[int, int]] = self._block_axes

        x = idctn(x, axes=block_axes, norm="ortho", type=2) + dc

        return x
