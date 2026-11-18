"""Unit checks of fusion, advection, the model and the losses on small synthetic inputs (CPU, a few seconds).

    python tests/test_core.py          (or: python -m pytest tests)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapixflow.flow import advect, advect_trajectory, substep_displacements  # noqa: E402
from adapixflow.fusion import OBJ, OCC, FusedScene, fuse_scene, object_points  # noqa: E402
from adapixflow.losses import distance_preservation, neighbors, relative_chamfer  # noqa: E402
from adapixflow.model import VelocityUNet, make_context  # noqa: E402


def _synthetic_cache(path, W=256, T=4):
    """One camera at the origin looking along -z at a wall 1 m away; a square in the middle is a moving object that
    shifts by 4 pixels per frame."""
    import io
    f = W / 2.0  # 90 degree field of view
    depth = np.ones((1, T, W, W), np.float16)
    seg = np.zeros((1, T, W, W), np.uint8)
    for t in range(T):
        seg[0, t, 96:160, 80 + 4 * t:144 + 4 * t] = 200
    c2w = np.tile(np.eye(4), (1, T, 1, 1))
    parts, offs = [], [0]
    for t in range(T):
        buf = io.BytesIO()
        Image.fromarray(np.full((W, W, 3), 128, np.uint8)).save(buf, format="JPEG")
        parts.append(np.frombuffer(buf.getvalue(), np.uint8))
        offs.append(offs[-1] + len(parts[-1]))
    meta = {"scene": path.stem, "moving_labels_override": [200]}
    np.savez(path, rgb_jpeg=np.concatenate(parts), rgb_offsets=np.array(offs, np.int64), depth=depth, seg=seg, c2w=c2w,
             f=np.float64(f), W=np.int64(W), meta=np.array(json.dumps(meta)))


def test_fusion_occupancy_and_unseen():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synthetic_cache(d / "toy.npz")
        msg = fuse_scene(d / "toy.npz", d / "fused", voxel=0.02, max_extent=4.0)
        assert msg.startswith("ok"), msg
        S = FusedScene(d / "fused", "toy")
        Z = S.crop([0], S.G // 2, S.G)[0][0]
        occ, obj = Z[OCC] > 0.5, (Z[OCC] > 0.5) & (Z[OBJ] > 0.5)
        assert obj.sum() > 0, "the moving square must produce object voxels"
        zw = S.lo[2] + (np.argwhere(obj)[:, 2] + 0.5) * S.voxel  # world z of object voxels
        assert np.all(np.abs(zw + 1.0) < 2 * S.voxel), "object voxels must lie on the wall at z = -1"
        # voxels behind the wall (further along -z) are unseen; voxels in front of it are observed free space
        behind = (S.lo[2] + (np.arange(S.G[2]) + 0.5) * S.voxel) < -1.0 - S.voxel
        front = (S.lo[2] + (np.arange(S.G[2]) + 0.5) * S.voxel) > -1.0 + S.voxel
        assert Z[5][..., behind][~occ[..., behind]].mean() > 0.99
        assert Z[5][..., front][~occ[..., front]].mean() < 0.01


def test_advection_constant_field():
    G = (16, 16, 16)
    Vn = [torch.zeros(1, 3, *G) for _ in range(5)]
    for V in Vn:
        V[:, 0] = 1.0  # one voxel per step along x
    p0 = torch.tensor([[4.0, 5.0, 6.0], [8.0, 8.0, 8.0]])
    p3 = advect(Vn, p0, 0, 3, 8)
    assert torch.allclose(p3, p0 + torch.tensor([3.0, 0, 0]), atol=1e-5)
    back = advect(Vn, p3, 3, 0, 8)
    assert torch.allclose(back, p0, atol=1e-5)
    D = substep_displacements(Vn, 3, 4)
    traj = advect_trajectory(D, p0, 4, +1)
    assert torch.allclose(traj[-1], p3, atol=1e-5)


def test_model_zero_init_and_shapes():
    model = VelocityUNet(horizon=12, amp=False)
    z = torch.rand(1, 6, 16, 16, 16)
    ctx = make_context(z, z, None)
    with torch.no_grad():
        V = model(ctx, torch.zeros(1), torch.full((1,), 0.2))
    assert V.shape == (1, 39, 16, 16, 16) and float(V.abs().max()) == 0.0  # zero-initialized: no motion


def test_losses():
    q = torch.rand(50, 3) * 10
    tgt = q + torch.tensor([2.0, 0, 0])
    assert abs(float(relative_chamfer(q.clone(), tgt, q, crop=48)) - 1.0) < 1e-4  # static prediction: ratio 1
    assert float(relative_chamfer(tgt.clone(), tgt, q, crop=48)) < 1e-2  # perfect prediction (cdist is exact to ~1e-3)
    nb = neighbors(q, 6)
    assert float(distance_preservation(q + 3.0, q, nb)) < 1e-5  # a translation preserves distances


def test_object_points():
    Z = np.zeros((6, 4, 4, 4), np.float32)
    Z[OCC, 1, 2, 3] = 1
    Z[OBJ, 1, 2, 3] = 0.9
    Z[OCC, 0, 0, 0] = 1  # occupied but not an object
    assert object_points(Z).tolist() == [[1.0, 2.0, 3.0]]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
