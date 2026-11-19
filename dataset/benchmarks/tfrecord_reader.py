"""Minimal TFRecord + tf.train.Example reader (numpy only, no TensorFlow), for the Kubric MOVi TFDS shards.

TFRecord framing: uint64 length, uint32 masked CRC of the length, payload, uint32 masked CRC of the payload (CRCs not checked).
tf.train.Example = Features{ map<string, Feature> }; Feature = oneof { BytesList(1), FloatList(2), Int64List(3) }; floats and
int64s are packed. Values are returned lazily as ("bytes", [bytes, ...]) / ("float", np.float32 array) / ("int64", np.int64 array).
The data files are untrusted: this parses bytes only and never executes anything from them.
Usage: tfrecord_reader.py <shard.tfrecord>   (prints every key of the first example with its type and length)
"""

from __future__ import annotations

import struct
import sys

import numpy as np


def records(path):
    """Yield the raw payload of every record in a TFRecord file."""
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = fh.read(n)
            fh.read(4)
            yield data


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 128:
            return result, i
        shift += 7


def _fields(buf):
    """(field number, wire type, value) of a protobuf message; length-delimited values as memoryview slices."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fn, wt, v


def _packed_varints(mv):
    b = np.frombuffer(mv, np.uint8)
    if len(b) == 0:
        return np.zeros(0, np.int64)
    ends = np.flatnonzero(b < 128)
    starts = np.concatenate([[0], ends[:-1] + 1])
    grp = np.repeat(np.arange(len(ends)), ends - starts + 1)
    pos = np.arange(len(b)) - starts[grp]
    vals = (b & 0x7F).astype(np.uint64) << (7 * pos).astype(np.uint64)
    return np.add.reduceat(vals, starts).view(np.int64)


def _feature(mv):
    for fn, wt, v in _fields(mv):
        if fn == 1:  # BytesList
            return "bytes", [bytes(x) for f2, _, x in _fields(v) if f2 == 1]
        if fn == 2:  # FloatList (packed)
            parts = [np.frombuffer(x, "<f4") if w2 == 2 else np.frombuffer(x, "<f4") for f2, w2, x in _fields(v) if f2 == 1]
            return "float", np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if fn == 3:  # Int64List (packed)
            parts = [_packed_varints(x) if w2 == 2 else np.array([x], np.int64) for f2, w2, x in _fields(v) if f2 == 1]
            return "int64", np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return "empty", None


def parse_example(payload):
    """{key: (kind, values)} of one serialized tf.train.Example."""
    mv = memoryview(payload)
    out = {}
    for fn, _, features in _fields(mv):
        if fn != 1:
            continue
        for f2, _, entry in _fields(features):
            if f2 != 1:
                continue
            key, val = None, None
            for f3, _, x in _fields(entry):
                if f3 == 1:
                    key = bytes(x).decode()
                elif f3 == 2:
                    val = x
            out[key] = _feature(val) if val is not None else ("empty", None)
    return out


if __name__ == "__main__":
    for j, rec in enumerate(records(sys.argv[1])):
        if j == 0:
            ex = parse_example(rec)
            for k in sorted(ex):
                kind, v = ex[k]
                desc = (f"{len(v)} items, first {len(v[0]) if v else 0} B {v[0][:8] if v else b''}" if kind == "bytes"
                        else f"{len(v)} values, first {v[:6].tolist()}" if v is not None else "")
                print(f"{k:45s} {kind:6s} {desc}")
    print("records:", j + 1)
