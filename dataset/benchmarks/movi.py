"""Decode one Kubric MOVi TFDS example (tfrecord_reader.parse_example output) into numpy arrays.

Encodings (challenges/movi/movi_a.py in the official repo, checked on the downloaded shards):
  video          24 PNG (uint8 RGB)
  depth          24 PNG (uint16), value / 65535 * (max - min) + min with metadata/depth_range = per-video min / max;
                 per the README the distance from the camera centre (not the planar z); see euclid_to_planar
  segmentations  24 PNG (uint8): instance index + 1, 0 = background (dome floor / sky)
  camera         positions (T, 3), quaternions (T, 4) WXYZ (pyquaternion), focal_length / sensor_width (mm); Blender /
                 OpenGL camera (x right, y up, looking along -z); principal point at the image centre
  instances      positions (K, T, 3), quaternions (K, T, 4) WXYZ, velocities, angular_velocities, bboxes_3d (K, T, 8, 3),
                 image_positions (K, T, 2) in [0, 1] (x right, y down), visibility (K, T) pixels, mass, friction,
                 restitution, plus variant labels (shape / size / material / asset_id / scale / is_dynamic)
  events         collisions: instances (n, 2) (65535 = floor / background), frame, force, position (n, 3), contact_normal
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfrecord_reader import parse_example, records  # noqa: E402


def quat_to_R(q):
    """(..., 4) WXYZ quaternions -> (..., 3, 3) rotation matrices (pyquaternion's convention)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _png(b, mode=None):
    im = Image.open(io.BytesIO(b))
    return np.asarray(im if mode is None else im.convert(mode))


def load(shard, index=None, name=None):
    """The example at position `index` of a shard (or with metadata/video_name == name), decoded."""
    for j, rec in enumerate(records(shard)):
        ex = parse_example(rec)
        vn = ex["metadata/video_name"][1][0].decode()
        if (index is not None and j == index) or (name is not None and vn == name):
            return decode(ex)
    raise KeyError(f"{index=} {name=} not in {shard}")


def list_names(shard):
    return [parse_example(rec)["metadata/video_name"][1][0].decode() for rec in records(shard)]


def decode(ex):
    g = lambda k: ex[k][1]  # noqa: E731
    T, H, W, K = int(g("metadata/num_frames")[0]), int(g("metadata/height")[0]), int(g("metadata/width")[0]), int(g("metadata/num_instances")[0])
    dmin, dmax = g("metadata/depth_range")
    out = {"name": g("metadata/video_name")[0].decode(), "T": T, "H": H, "W": W, "K": K}
    out["rgb"] = np.stack([_png(b, "RGB") for b in g("video")])
    out["depth_raw"] = np.stack([_png(b).astype(np.float64) for b in g("depth")]).reshape(T, H, W) / 65535 * (dmax - dmin) + dmin
    out["seg"] = np.stack([_png(b) for b in g("segmentations")]).reshape(T, H, W).astype(np.uint8)
    out["rgb_png"] = list(g("video"))
    cam = {"focal_length": float(g("camera/focal_length")[0]), "sensor_width": float(g("camera/sensor_width")[0]),
           "field_of_view": float(g("camera/field_of_view")[0]),
           "positions": g("camera/positions").reshape(T, 3).astype(np.float64),
           "quaternions": g("camera/quaternions").reshape(T, 4).astype(np.float64)}
    c2w = np.tile(np.eye(4), (T, 1, 1))
    c2w[:, :3, :3] = quat_to_R(cam["quaternions"])
    c2w[:, :3, 3] = cam["positions"]
    cam["c2w"] = c2w
    cam["f_px"] = cam["focal_length"] / cam["sensor_width"] * W  # square sensor (sensor_height = sensor_width * H / W)
    out["camera"] = cam
    inst = {}
    for k in ("positions", "velocities", "angular_velocities"):
        inst[k] = g(f"instances/{k}").reshape(K, T, 3).astype(np.float64)
    inst["quaternions"] = g("instances/quaternions").reshape(K, T, 4).astype(np.float64)
    inst["bboxes_3d"] = g("instances/bboxes_3d").reshape(K, T, 8, 3).astype(np.float64)
    inst["image_positions"] = g("instances/image_positions").reshape(K, T, 2)
    inst["visibility"] = g("instances/visibility").reshape(K, T)
    for k in ("mass", "friction", "restitution", "scale"):
        if f"instances/{k}" in ex:
            inst[k] = g(f"instances/{k}").astype(np.float64)
    for k in ("shape_label", "size_label", "material_label", "color_label", "is_dynamic", "category"):
        if f"instances/{k}" in ex:
            inst[k] = g(f"instances/{k}")
    if "instances/asset_id" in ex:
        inst["asset_id"] = [b.decode() for b in g("instances/asset_id")]
    out["instances"] = inst
    n = len(g("events/collisions/frame"))
    out["collisions"] = {"instances": g("events/collisions/instances").reshape(n, 2), "frame": g("events/collisions/frame"),
                         "force": g("events/collisions/force"), "position": g("events/collisions/position").reshape(n, 3),
                         "contact_normal": g("events/collisions/contact_normal").reshape(n, 3)}
    out["background"] = g("background")[0].decode() if "background" in ex else None
    return out


def euclid_to_planar(d, f, W, H=None):
    """Distance from the camera centre (Kubric's depth) -> planar depth z along -z (our caches), pixel centres at +0.5."""
    H = W if H is None else H
    v, u = np.mgrid[0:H, 0:W] + 0.5
    return d / np.sqrt(1 + ((u - W / 2) / f) ** 2 + ((v - H / 2) / f) ** 2)


def unproject(z, c2w, f, W, mask=None):
    """World points of planar depth z (H, W) with an OpenGL c2w (as the cache / lift: pc = [(u+.5-W/2)/f*z, -(v+.5-W/2)/f*z, -z])."""
    H = z.shape[0]
    v, u = np.mgrid[0:H, 0:W] + 0.5
    pc = np.stack([(u - W / 2) / f * z, -(v - H / 2) / f * z, -z], -1)
    if mask is not None:
        pc = pc[mask]
    return pc.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3]


def project(P, c2w, f, W, H=None):
    """Pixel coordinates (u, v) (x right, y down; pixel centres at +0.5) and planar depth of world points P (n, 3)."""
    H = W if H is None else H
    pc = (P - c2w[:3, 3]) @ c2w[:3, :3]
    z = -pc[:, 2]
    return np.stack([f * pc[:, 0] / z + W / 2, -f * pc[:, 1] / z + H / 2], 1), z
