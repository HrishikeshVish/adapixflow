# Third-party code

Pinned submodules, at the commits behind the paper. None is needed to train or evaluate AdaPixFlow.

| Path | Used for | Upstream |
| :--- | :--- | :--- |
| `Depth-Anything-3` | estimated depth (`dataset/estimated/da3_depth.py`) | https://github.com/ByteDance-Seed/Depth-Anything-3 |
| `FreeGave` | baseline (per-scene fit) | https://github.com/vLAR-group/FreeGave |
| `TRACE` | baseline (per-scene fit) | https://github.com/vLAR-group/TRACE |
| `shape-of-motion` | baseline (per-scene fit) | https://github.com/vye16/shape-of-motion |
| `MoSca` | baseline (per-scene fit) | https://github.com/JiahuiLei/MoSca |
| `SpaTrackerV2` | baseline (feed-forward) | https://github.com/henry123-boy/SpaTrackerV2 |
| `Open-d4rt` | baseline (feed-forward, unofficial D4RT) | https://github.com/Lijiaxin0111/Open-d4rt |

SAM 2.1 (https://github.com/facebookresearch/sam2) runs through its `transformers` implementation with the weights
`facebook/sam2.1-hiera-large`, so it needs no submodule.

```bash
git submodule update --init --recursive                      # all of them, or name one path
git -C third_party/MoSca apply ../patches/MoSca.patch        # likewise FreeGave, TRACE, shape-of-motion
```

`patches/` holds our changes to four of them: build fixes for newer CUDA and compilers, NumPy 2 and gsplat >= 1.5
compatibility, a scikit-learn fallback where cuML is unavailable, the MoSca `camera.py` symlink, and an opt-in switch
in Shape of Motion (`SOM_MASK_INVALID_DEPTH=1`) that leaves pixels without depth out of its depth losses.
