# SPDX-License-Identifier: GPL-3.0-or-later
#
# fetch.py -- fetch video
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import ffmpeg
import numpy as np

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    frame_rate: float
    frames: int
    height: int
    width: int


def frames_external_format(
    path: Path,
    metadata: VideoMetadata,
) -> Iterator[np.ndarray]:
    from subprocess import (
        DEVNULL,
        PIPE,
        Popen,
    )

    args: list[str] = (
        ffmpeg.input(path)
        .output("pipe:", format="rawvideo", pix_fmt="gray")
        .compile()
    )

    process: Popen = Popen(args, stdin=DEVNULL, stdout=PIPE, stderr=DEVNULL)

    frame_shape: tuple[int, int] = (metadata.height, metadata.width)
    frame_bytes: int = metadata.height * metadata.width

    while True:
        in_bytes = process.stdout.read(frame_bytes)

        if not len(in_bytes):
            break

        assert len(in_bytes) == frame_bytes

        frame = np.frombuffer(in_bytes, np.uint8).reshape(frame_shape)

        yield frame

    process.stdout.close()
    process.wait()


def probe_external_format(
    path: Path,
    ffmpeg_args: dict[str, str] | None = None,
) -> VideoMetadata:
    from fractions import Fraction

    if ffmpeg_args is None:
        ffmpeg_args = {}

    probe: dict[str, dict] = ffmpeg.probe(path, **ffmpeg_args)

    stream: dict = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )

    frame_rate: float = float(Fraction(stream["r_frame_rate"]))
    frames: int = int(float(stream["duration"]) * frame_rate)

    return VideoMetadata(
        frame_rate=frame_rate,
        frames=frames,
        height=stream["height"],
        width=stream["width"],
    )
