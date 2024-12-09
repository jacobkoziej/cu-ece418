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
    Callable,
    Iterator,
    Optional,
)

from PIL import Image
from tqdm import trange

from decode import (
    Decoder,
    DecoderConfig,
)
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
from serialize import (
    read,
    write,
)


def encode(args: Namespace, ffmpeg_args: dict[str, str]):
    metadata: VideoMetadata = probe_external_format(args.input, ffmpeg_args)
    frame_fetcher: Iterator[np.ndarray] = frames_external_format(
        args.input, metadata
    )

    stream_config: StreamConfig = StreamConfig(
        block_size=args.block_size,
        quality=args.quality,
        reference_frames_max=args.reference_frames_max,
    )

    frame_rate_config: EncoderFrameRateConfig = EncoderFrameRateConfig(
        i=args.i_frame_rate,
        p=args.p_frame_rate,
    )
    encoder_config: EncoderConfig = EncoderConfig(
        frame_rate=frame_rate_config,
        stream=stream_config,
        metadata=metadata,
        search_limit=args.block_search_limit,
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


def decode(args: Namespace, ffmpeg_args: dict[str, str]):
    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.input, "rb") as f:
        metadata: VideoMetadata = read(f)
        assert isinstance(metadata, VideoMetadata)

        stream_config: StreamConfig = read(f)
        assert isinstance(stream_config, StreamConfig)

        decoder_config: DecoderConfig = DecoderConfig(
            stream=stream_config,
            metadata=metadata,
        )

        decoder: Decoder = Decoder(decoder_config)

        digits: int = len(str(metadata.frames))

        def save_frame(frame_index: int, frame: np.ndarray):
            file_name = f"{args.output}/{str(frame_index).zfill(digits)}.png"

            frame_out: Image = Image.fromarray(frame)

            frame_out.save(file_name)

        frame: Optional[np.ndarray] = None
        frame_index: int = -1

        for _ in trange(metadata.frames):
            decoded_frame: FrameType = read(f)

            frame = decoder.step(decoded_frame)

            if frame is None:
                continue

            frame_index += 1
            save_frame(frame_index, frame)

        frames: list[np.ndarray] = decoder.flush_decode_buffer()

        for frame in frames:
            frame_index += 1
            save_frame(frame_index, frame)


def main():
    parser: ArgumentParser = ArgumentParser(
        description="video compressor",
    )

    parser.add_argument(
        "--block-search-limit",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--block-size",
        default=16,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--decode-buffer-size",
        default=32,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--i-frame-rate",
        default=32,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--p-frame-rate",
        default=8,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--quality",
        default=16,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--reference-frames-max",
        default=2,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--step-size",
        default=16,
        required=False,
        type=int,
    )

    args: Namespace
    ffmpeg_args: list[str]
    args, ffmpeg_args = parser.parse_known_args()

    ffmpeg_args: dict[str, str] = dict(arg.split("=") for arg in ffmpeg_args)

    operation: Callable[Namespace, dict[str, str]] = (
        decode if args.input.suffix == ".cu-ece418" else encode
    )

    operation(args, ffmpeg_args)


if __name__ == "__main__":
    main()
