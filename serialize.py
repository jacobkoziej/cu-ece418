# SPDX-License-Identifier: GPL-3.0-or-later
#
# serialize.py -- serialize stream
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import msgpack

from dataclasses import asdict
from typing import Any

from fetch import VideoMetadata
from frame import StreamConfig


def _decode(obj: Any) -> Any:
    if (k := str(StreamConfig)) in obj:
        obj.pop(k)

        return StreamConfig(**obj)

    if (k := str(VideoMetadata)) in obj:
        obj.pop(k)

        return VideoMetadata(**obj)

    return obj


def _encode(obj: Any) -> Any:
    match obj:
        case StreamConfig() | VideoMetadata():
            return asdict(obj) | {str(type(obj)): True}

    return obj


def decode(stream: bytes) -> Any:
    return msgpack.unpackb(stream, object_hook=_decode)


def encode(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode)
