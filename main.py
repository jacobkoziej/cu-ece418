#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# main.py -- video compressor
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>


def main():
    from argparse import (
        ArgumentParser,
        Namespace,
    )
    from pathlib import Path
    from sys import argv

    parser = ArgumentParser(
        prog=argv[0],
        description="video compressor",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
    )

    args: Namespace
    ffmpeg_args: list[str]
    args, ffmpeg_args = parser.parse_known_args()

    ffmpeg_args: dict[str, str] = dict(arg.split("=") for arg in ffmpeg_args)


if __name__ == "__main__":
    main()
