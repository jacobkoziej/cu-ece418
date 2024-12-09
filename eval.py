#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# eval.py -- evaluate compression
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import numpy as np

from argparse import (
    ArgumentParser,
    Namespace,
)
from pathlib import Path

from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)
from tqdm.contrib import tzip


def main():
    parser: ArgumentParser = ArgumentParser(
        description="compression evaluator",
    )

    parser.add_argument(
        "-c",
        "--compressed",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        type=Path,
    )

    args: Namespace = parser.parse_args()

    source: list[Path] = list(args.source.glob("*.png"))
    compressed: list[Path] = list(args.compressed.glob("*.png"))

    source.sort()
    compressed.sort()

    assert len(source) == len(compressed)

    psnr: list[float] = []
    ssim: list[float] = []

    for s, c in tzip(source, compressed):
        s: np.ndarray = np.array(Image.open(s).convert("L"))
        c: np.ndarray = np.array(Image.open(c).convert("L"))

        psnr += [peak_signal_noise_ratio(s, c)]
        ssim += [structural_similarity(s, c)]

    psnr: np.ndarray = np.array(psnr)
    ssim: np.ndarray = np.array(ssim)

    print(
        f"psnr:\t{psnr.mean()} (avg)\t {psnr.min()} (min)\t {psnr.max()} (max)"
    )
    print(
        f"ssim:\t{ssim.mean()} (avg)\t {ssim.min()} (min)\t {ssim.max()} (max)"
    )


if __name__ == "__main__":
    main()
