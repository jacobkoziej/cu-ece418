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
    FrameType,
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


class Decoder:
    def __init__(self, config: DecoderConfig):
        self.config: DecoderConfig = config

        height: int = config.metadata.height
        width: int = config.metadata.width
        block_size: int = config.stream.block_size

        self._unpad_frame: Callable[[np.ndarray], np.ndarray] = (
            self.unpad_frame(pad_width(height, width, block_size))
        )

        self._decode_buffer = deque(maxlen=config.decode_buffer_size)
        self._reference_buffer = deque(
            maxlen=config.stream.reference_frames_max
        )

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

                block: tuple[int, slice, slice] = select_block(
                    i, j, motion_vector
                )

                reference_frame: np.ndarray = reference_frames[block]

                y = block2pel(i)
                x = block2pel(j)

                write_block: tuple[slice, slice] = (
                    slice(y, y + block_size),
                    slice(x, x + block_size),
                )

                decoded_frame[write_block] = reference_frame + residual

        return decoded_frame

    def reference_frames(self, x: int) -> np.ndarray:
        assert x >= 0

        reference_buffer: deque = self._reference_buffer

        reference_frames: np.ndarray = np.array(
            [reference_buffer[-i] for i in range(x, 0, -1)]
        )

        return reference_frames

    def step(self, frame: FrameType) -> None:
        decoded_frame: np.ndarray
        reference_frames: np.ndarray

        decode_buffer: deque = self._decode_buffer
        reference_buffer: deque = self._reference_buffer

        match frame:
            case BFrame() | PFrame():
                reference_frames = self.reference_frames(
                    frame.reference_frames
                )

                decoded_frame = self.decode_frame(
                    reference_frames,
                    frame.motion_vectors,
                    frame.residuals,
                )

            case IFrame():
                decoded_frame = frame.frame

            case _:
                return None

        if len(decode_buffer) >= decode_buffer.maxlen:
            _ = decode_buffer.popleft()

        insert_index: int = len(decode_buffer) - 1

        if not isinstance(frame, BFrame):
            if len(reference_buffer) >= reference_buffer.maxlen:
                _ = reference_buffer.popleft()

            reference_buffer.append(decoded_frame)

            insert_index += 1

        decode_buffer.insert(insert_index, decoded_frame)

    @staticmethod
    def unpad_frame(
        pad_width: tuple[tuple[int, int], tuple[int, int]]
    ) -> Callable[[np.ndarray], np.ndarray]:
        precomputed_unpad_slice = unpad_slice(pad_width)

        def unpad_frame(x: np.ndarray) -> np.ndarray:
            unpad_slice = precomputed_unpad_slice

            return x[..., *unpad_slice]

        return unpad_frame
