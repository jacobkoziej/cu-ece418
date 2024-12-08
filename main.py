#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# main.py -- video compressor
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from argparse import (
    ArgumentParser,
    Namespace,
)
from pathlib import Path
from typing import (
    Iterator,
    Optional,
)

from tqdm import trange

from encode import (
    Encoder,
    EncoderConfig,
    EncoderFrameRateConfig,
)
from fetch import (
    VideoMetadata,
    frames_external_format,
    probe_external_format,
)
from frame import (
    FrameType,
    StreamConfig,
)
from serialize import write


def main():
    parser: ArgumentParser = ArgumentParser(
        description="video compressor",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--step-size",
        default=8,
        required=False,
        type=int,
    )

    args: Namespace
    ffmpeg_args: list[str]
    args, ffmpeg_args = parser.parse_known_args()

    ffmpeg_args: dict[str, str] = dict(arg.split("=") for arg in ffmpeg_args)

    metadata: VideoMetadata = probe_external_format(args.input, ffmpeg_args)
    frame_fetcher: Iterator[np.ndarray] = frames_external_format(
        args.input, metadata
    )

    stream_config: StreamConfig = StreamConfig()

    frame_rate_config: EncoderFrameRateConfig = EncoderFrameRateConfig()
    encoder_config: EncoderConfig = EncoderConfig(
        frame_rate=frame_rate_config,
        stream=stream_config,
        metadata=metadata,
    )

    encoder: Encoder = Encoder(encoder_config, frame_fetcher)

    with open(args.output, "wb") as f:
        write(f, metadata)
        write(f, stream_config)

        for _ in trange(0, metadata.frames, args.step_size):
            frames: Optional[list[FrameType]] = encoder.step(args.step_size)

            if frames is None:
                continue

            for frame in frames:
                write(f, frame)


if __name__ == "__main__":
    main()
