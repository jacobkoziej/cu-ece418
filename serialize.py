# SPDX-License-Identifier: GPL-3.0-or-later
#
# serialize.py -- serialize stream
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import struct

import msgpack
import msgpack_numpy as msgpack_np
import numpy as np

from dataclasses import asdict
from io import BufferedWriter
from typing import Any

from fetch import VideoMetadata
from frame import (
    BFrame,
    IFrame,
    PFrame,
    StreamConfig,
)
from quantize import QuantizedValues


def _decode(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return msgpack.unpackb(obj, object_hook=msgpack_np.decode)

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _decode(v)

    else:
        return obj

    if (k := str(BFrame)) in obj:
        obj.pop(k)

        return BFrame(**obj)

    if (k := str(IFrame)) in obj:
        obj.pop(k)

        return IFrame(**obj)

    if (k := str(PFrame)) in obj:
        obj.pop(k)

        return PFrame(**obj)

    if (k := str(StreamConfig)) in obj:
        obj.pop(k)

        return StreamConfig(**obj)

    if (k := str(QuantizedValues)) in obj:
        obj.pop(k)

        return QuantizedValues(**obj)

    if (k := str(VideoMetadata)) in obj:
        obj.pop(k)

        return VideoMetadata(**obj)

    return obj


def _encode(obj: Any) -> Any:
    match obj:
        case (
            BFrame()
            | IFrame()
            | PFrame()
            | QuantizedValues()
            | StreamConfig()
            | VideoMetadata()
        ):
            obj = asdict(obj) | {str(type(obj)): True}

            for k, v in obj.items():
                obj[k] = _encode(v)

        case np.ndarray():
            obj = msgpack.packb(obj, default=msgpack_np.encode)

    return obj


def decode(stream: bytes) -> Any:
    return msgpack.unpackb(stream, object_hook=_decode)


def encode(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode)


def write(stream: BufferedWriter, obj: Any) -> int:
    b: bytes = encode(obj)

    count: int = 0

    count += stream.write(struct.pack("<L", len(b)))
    count += stream.write(b)

    return count
