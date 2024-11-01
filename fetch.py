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


def probe_external_format(
    path: Path,
    ffmpeg_args: dict[str, str] | None = None,
) -> VideoMetadata:
    import ffmpeg

    from fractions import Fraction

    if ffmpeg_args is None:
        ffmpeg_args = {}

    probe: dict[str, dict] = ffmpeg.probe(path, **ffmpeg_args)

    stream: dict = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )

    frame_rate: float = float(Fraction(stream["r_frame_rate"]))

    return VideoMetadata(
        frame_rate=frame_rate,
        height=stream["height"],
        width=stream["width"],
    )
