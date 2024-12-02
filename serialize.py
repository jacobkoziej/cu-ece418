# SPDX-License-Identifier: GPL-3.0-or-later
#
# serialize.py -- serialize stream
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import msgpack

from typing import Any


def _decode(obj: Any) -> Any:
    return obj


def _encode(obj: Any) -> Any:
    return obj


def decode(stream: bytes) -> Any:
    return msgpack.unpackb(stream, object_hook=_decode)


def encode(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode)
