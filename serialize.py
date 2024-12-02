# SPDX-License-Identifier: GPL-3.0-or-later
#
# serialize.py -- serialize stream
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>

import msgpack

from io import (
    BufferedReader,
    BufferedWriter,
)
from typing import Any


def _decode(obj: Any) -> Any:
    return obj


def _encode(obj: Any) -> Any:
    return obj


def decode(stream: BufferedReader) -> Any:
    return msgpack.unpack(stream, object_hook=_decode)


def encode(stream: BufferedWriter, obj: Any) -> None:
    msgpack.pack(obj, stream, default=_encode)
