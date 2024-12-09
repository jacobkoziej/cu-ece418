# SPDX-License-Identifier: GPL-3.0-or-later
#
# encode.py -- encode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Final,
    Iterator,
    Optional,
)

from decode import (
    Decoder,
    DecoderConfig,
)
from fetch import VideoMetadata
from frame import (
    BFrame,
    FrameType,
    IFrame,
    PFrame,
    StreamConfig,
    pad_width,
)
from quantize import (
    Magnitude,
    Quantizer,
)


@dataclass
class EncoderFrameRateConfig:
    i: int = 32
    p: int = 8


@dataclass
class EncoderConfig:
    frame_rate: EncoderFrameRateConfig
    metadata: VideoMetadata
    stream: StreamConfig
    search_limit: int = 1


class Encoder:
    def __init__(self, config: EncoderConfig, stream: Iterator[np.ndarray]):
        self.config: EncoderConfig = config

        height: int = config.metadata.height
        width: int = config.metadata.width
        block_size: int = config.stream.block_size

        self._pad_frame: Callable[[np.ndarray], np.ndarray] = self.pad_frame(
            height, width, block_size
        )

        self._stream: Iterator[np.ndarray] = stream

        decoder_config: DecoderConfig = DecoderConfig(
            stream=config.stream,
            metadata=config.metadata,
        )
        self._decoder: Decoder = Decoder(decoder_config)

        block_size: int = config.stream.block_size
        block_shape: tuple[int, int] = (block_size, block_size)
        self._quantizer: Magnitude = Magnitude(
            quality=config.stream.quality,
            block_shape=block_shape,
        )

        self._current_frame: int = -1
        self._bframe_queue: deque = deque()

    def _step(self, x: np.ndarray) -> list[FrameType]:
        assert x.ndim == 3

        bframe_queue: Final[deque] = self._bframe_queue
        config: Final[EncoderConfig] = self.config
        decoder: Final[Decoder] = self._decoder
        quantizer: Final[Quantizer] = self._quantizer

        reference_frames: np.ndarray
        motion_vectors: np.ndarray
        residuals: np.ndarray

        self._current_frame += 1

        frame: Optional[FrameType] = None

        iframe: bool = not self._current_frame % config.frame_rate.i
        pframe: bool = not self._current_frame % config.frame_rate.p or (
            self._current_frame == config.metadata.frames - 1
        )

        if iframe:
            frame = IFrame(x.squeeze())

        elif pframe:
            reference_frames = decoder.reference_frames(1)
            motion_vectors, residuals = self.block_match(reference_frames, x)

            frame = PFrame(
                motion_vectors=motion_vectors,
                reference_frames=1,
                residuals=quantizer.quantize(residuals),
            )

        if not frame:
            bframe_queue.append(x)

            return []

        _ = decoder.step(frame)

        frames: list[np.ndarray] = [frame]

        for x in bframe_queue:
            reference_frames = decoder.reference_frames(2)
            motion_vectors, residuals = self.block_match(reference_frames, x)

            frame = BFrame(
                motion_vectors=motion_vectors,
                reference_frames=2,
                residuals=quantizer.quantize(residuals),
            )
            frames += [frame]

            decoder.step(frame)

        bframe_queue.clear()

        return frames

    def block_match(
        self, search: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert search.ndim == 3
        assert target.ndim == 3

        assert target.shape[-2:] == search.shape[-2:]
        assert target.dtype == search.dtype

        height: int
        width: int
        _, height, width = target.shape

        block_size: int = self.config.stream.block_size

        assert not height % block_size
        assert not width % block_size

        blocked_height: int = height // block_size
        blocked_width: int = width // block_size

        search_limit: int = self.config.search_limit
        search_range: int = block_size * search_limit

        motion_vectors: np.ndarray = np.zeros(
            shape=(blocked_height, blocked_width, 3),
            dtype=np.int32,
        )
        residuals: np.ndarray = np.zeros(
            shape=(blocked_height, blocked_width) + (block_size,) * 2,
            dtype=np.int8,
        )

        def range_process(limit: int) -> range:
            return range(0, limit - block_size + 1, block_size)

        def range_search(position: int, limit: int) -> range:
            return range(
                max(0, position - search_range),
                min(limit - block_size + 1, position + search_range + 1),
                block_size,
            )

        def select_block(x: int, y: int) -> tuple[slice, slice]:
            block = (
                slice(x, x + block_size),
                slice(y, y + block_size),
            )

            return block

        def pel2block(x: int) -> int:
            return x // block_size

        def select_output_block(x: int, y: int) -> tuple[slice, slice]:
            output_block = (
                slice(x, x + block_size),
                slice(y, y + block_size),
            )

            return output_block

        def process_candidate(
            i: int,
            j: int,
            x: int,
            y: int,
            threshold: int,
        ) -> tuple[int, np.ndarray, np.ndarray]:
            candidate: np.ndarray = search[..., *select_block(y, x)]

            residual: np.ndarray = block - candidate

            sad: np.ndarray = np.sum(np.abs(residual), axis=(-2, -1))

            forward_prediction: int = int(np.argmin(sad))

            sad: int = sad[forward_prediction]

            residual = residual[forward_prediction]

            best_match: np.ndarray = np.array(
                [forward_prediction, pel2block(y - i), pel2block(x - j)]
            )

            return (sad, best_match, residual)

        def search_area(i, j, block) -> tuple[np.ndarray, np.ndarray]:
            sad: float = float("inf")

            for y in range_search(i, height):
                for x in range_search(j, width):
                    s, b, r = process_candidate(i, j, x, y, sad)

                    if s < sad:
                        sad = s
                        best_match = b
                        residual = r

            return (best_match, residual)

        for i in range_process(height):
            for j in range_process(width):
                block: np.ndarray = target[..., *select_block(i, j)]

                best_match, residual = search_area(i, j, block)

                block_i = pel2block(i)
                block_j = pel2block(j)

                motion_vectors[block_i, block_j] = best_match
                residuals[block_i, block_j] = residual

        assert np.all((motion_vectors >= -128) & (motion_vectors <= 127))

        motion_vectors = motion_vectors.astype(np.int8)

        return (motion_vectors, residuals)

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

    def step(self, steps: int) -> list[np.ndarray]:
        assert steps >= 1

        config: Final[EncoderConfig] = self.config
        stream: Final[Iterator[np.ndarray]] = self._stream

        encoded_frames: list[np.ndarray] = []

        for step in range(steps):
            if self._current_frame >= config.metadata.frames:
                break

            frame: np.ndarray
            frame = self._pad_frame(next(stream))
            frame = np.expand_dims(frame, 0)

            encoded_frames += self._step(frame)

        return encoded_frames
