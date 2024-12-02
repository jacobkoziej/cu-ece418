# SPDX-License-Identifier: GPL-3.0-or-later
#
# serialize.py -- serialize stream
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import msgpack
import msgpack_numpy as msgpack_np
import numpy as np

from dataclasses import asdict
from typing import Any

from fetch import VideoMetadata
from frame import (
    IFrame,
    StreamConfig,
)


def _decode(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return msgpack.unpackb(obj, object_hook=msgpack_np.decode)

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _decode(v)

    else:
        return obj

    if (k := str(IFrame)) in obj:
        obj.pop(k)

        return IFrame(**obj)

    if (k := str(StreamConfig)) in obj:
        obj.pop(k)

        return StreamConfig(**obj)

    if (k := str(VideoMetadata)) in obj:
        obj.pop(k)

        return VideoMetadata(**obj)

    return obj


def _encode(obj: Any) -> Any:
    match obj:
        case IFrame() | StreamConfig() | VideoMetadata():
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
