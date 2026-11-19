"""Minimal reader for Spring-Gaus simulation PLYs (binary_little_endian, float vertex properties only). Data-only parsing:
no code from the dataset is executed."""

from __future__ import annotations

import numpy as np


def read_ply(path):
    """-> (props list, (n, len(props)) float32 array) for a binary little-endian PLY whose vertex properties are all float."""
    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            ch = fh.read(1)
            if not ch:
                raise ValueError(f"{path}: no end_header")
            head += ch
        lines = head.decode("ascii").splitlines()
        assert lines[0] == "ply" and "binary_little_endian" in lines[1], lines[:2]
        n, props = 0, []
        for ln in lines:
            t = ln.split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            elif t[:1] == ["property"]:
                assert t[1] == "float", ln
                props.append(t[2])
        data = np.fromfile(fh, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    return props, data
