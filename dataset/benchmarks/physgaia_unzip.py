"""Extract one PhysGaia scene from its release zip into <out>/<scene>/ (camera_info_*.json, render/, particles/), dropping
the archive's leading directories. The zip is untrusted data: member paths are checked to stay inside the output, and
nothing in it is run.

Usage: python -I dataset/benchmarks/physgaia_unzip.py <scene.zip> <scene> <out dir>
"""

import shutil
import sys
import zipfile
from pathlib import Path

src, scene, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).resolve()
dst = out / scene
n = 0
with zipfile.ZipFile(src) as z:
    for zi in z.infolist():
        parts = zi.filename.split("/")
        if scene not in parts or zi.filename.endswith("/"):
            continue
        rel = Path(*parts[parts.index(scene) + 1:])
        target = (dst / rel).resolve()
        assert dst.resolve() in target.parents, f"unsafe member {zi.filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(zi) as f, open(target, "wb") as g:
            shutil.copyfileobj(f, g)
        n += 1
print(f"extracted {n} files of {scene} into {dst}")
