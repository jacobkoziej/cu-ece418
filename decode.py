# SPDX-License-Identifier: GPL-3.0-or-later
#
# decode.py -- decode video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fetch import VideoMetadata
from frame import (
    BFrame,
    IFrame,
    PFrame,
    StreamConfig,
    pad_width,
    unpad_slice,
)


@dataclass
class DecoderConfig:
    metadata: VideoMetadata
    stream: StreamConfig
    decode_buffer_size: int = 32
    reference_buffer_size: int = 2


class Decoder:
    def __init__(self, config: DecoderConfig):
        self.config: DecoderConfig = config

        height: int = config.metadata.height
        width: int = config.metadata.width
        block_size: int = config.stream.block_size

        self._unpad_frame: Callable[[np.ndarray], np.ndarray] = self.unpad_frame(
            pad_width(height, width, block_size)
        )

        self._decode_buffer = deque(maxlen=config.decode_buffer_size)
        self._reference_buffer = deque(maxlen=config.reference_buffer_size)

    def decode(
        self,
        frame: BFrame | IFrame | PFrame,
        *,
        unpad: bool = True,
    ) -> np.ndarray:
        decoded_frame: np.ndarray
        reference_frames: np.ndarray

        config: DecoderConfig = self.config
        decode_buffer: deque = self._decode_buffer
        reference_buffer: deque = self._reference_buffer

        def get_reference_frames(x: int) -> np.ndarray:
            reference_frames: np.ndarray = np.array(
                [reference_buffer[-i] for i in range(x + 1, 0, -1)]
            )

            return reference_frames

        match frame:
            case BFrame() | PFrame():
                reference_frames = get_reference_frames(frame.reference_frame)

                decoded_frame = self.decode_frame(
                    reference_frames,
                    frame.motion_vectors,
                    frame.residuals,
                )

            case IFrame():
                decoded_frame = frame.frame

            case _:
                return None

        if len(decode_buffer) >= config.decode_buffer_size:
            _ = decode_buffer.popleft()

        if len(reference_buffer) >= config.reference_buffer_size:
            _ = reference_buffer.popleft()

        if not isinstance(frame, BFrame):
            reference_buffer.append(decoded_frame)

        decode_buffer.append(decoded_frame)

        if unpad:
            decoded_frame = self._unpad_frame(decoded_frame)

        return decoded_frame

    def decode_frame(
        self,
        reference_frames: np.ndarray,
        motion_vectors: np.ndarray,
        residuals: np.ndarray,
    ) -> np.ndarray | None:
        blocks_i: int
        blocks_j: int
        height: int
        width: int
        blocks_i, blocks_j, height, width = residuals.shape

        block_size: int = self.config.stream.block_size

        assert not height % block_size
        assert not width % block_size

        assert reference_frames.dtype == residuals.dtype

        decoded_frame: np.ndarray = np.zeros(
            shape=reference_frames.shape[-2:],
            dtype=reference_frames.dtype,
        )

        def block2pel(x: int) -> int:
            return x * block_size

        def select_block(
            i: int,
            j: int,
            motion_vector: np.ndarray,
        ) -> tuple[int, slice, slice]:
            reference_frame: int
            x: int
            y: int
            reference_frame, y, x = motion_vector

            y = block2pel(y + i)
            x = block2pel(x + j)

            block = (
                reference_frame,
                slice(y, y + block_size),
                slice(x, x + block_size),
            )

            return block

        for i in range(blocks_i):
            for j in range(blocks_j):
                motion_vector: np.ndarray = motion_vectors[i, j]
                residual: np.ndarray = residuals[i, j]

                block: tuple[int, slice, slice] = select_block(i, j, motion_vector)

                reference_frame: np.ndarray = reference_frames[*block]

                decoded_frame[*block[1:]] = reference_frame + residual

        return decoded_frame

    @staticmethod
    def unpad_frame(
        pad_width: tuple[tuple[int, int], tuple[int, int]]
    ) -> Callable[[np.ndarray], np.ndarray]:
        precomputed_unpad_slice = unpad_slice(pad_width)

        def unpad_frame(x: np.ndarray) -> np.ndarray:
            unpad_slice = precomputed_unpad_slice

            return x[..., *unpad_slice]

        return unpad_frame
