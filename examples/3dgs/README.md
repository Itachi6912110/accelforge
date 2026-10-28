# 3DGS alpha compute / alpha blend on AccelForge

Unfused per-stage models of the 3D Gaussian Splatting rasterizer's per-pixel stages
(`renderCUDA`): alpha computation and front-to-back alpha blending, each on a
DRAM -> GlobalBuffer (cache) -> 16x16 PE-array architecture costed with hwcomponents
models. Defaults are calibrated to one frame (garden, 960x622, 2340 16x16 tiles,
1.74M Gaussian-tile pairs). The write-up, results, and modeling limitations live in
`plans/3dgs-accelforge/` of the 3DGS-Arch-Evaluation repo.

| file | what |
|---|---|
| `workloads/alpha_compute.yaml` | `Alpha[q, k] = f(GaussParams[k])`, one tile, `n_instances` = tiles |
| `workloads/alpha_blend.yaml` | `PixelOut[q] (RGB+T) += Alpha[q, k] * Color[k]`, one tile |
| `workloads/*_stencil.yaml` | same, tiles as ranks, Gaussians binned so each bin feeds a 2x2 tile window (cross-tile reuse as an affine stencil) |
| `arches/alpha_compute_array.yaml` | LPDDR4 -> SimpleSRAMCache -> 16x16 IntMAC array (12 MAC-eq / alpha) |
| `arches/alpha_blend_array.yaml` | LPDDR4 -> SimpleSRAMCache -> 16x16 IntMAC array (6 MAC-eq / blend) |
| `dram_vs_sram_sweep.py` | maps every (variant, stage, GlobalBuffer size); CSVs, per-point mappings, figures |
| `probe_limitations.py` | tries to express each 3DGS feature (sparsity, indirection, ragged lists, transmittance scan, cross-tile reuse) and records accept / reject |

```bash
# from the accelforge repo root
python examples/3dgs/probe_limitations.py   --out <dir>
python examples/3dgs/dram_vs_sram_sweep.py  --out <dir>   # ~5 min, 17 sizes x 2 stages x 2 variants
```

`probe_limitations.py`'s ragged-list probe needs Barvinok-enabled ISL
(`pip install islpy-barvinok`, which replaces `islpy`); without it that probe is
rejected with "Non-box-shaped sets are not supported".
