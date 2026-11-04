# 3DGS render on AccelForge

Workload and architecture descriptors for the 3DGS render analysis defined in
`3DGSArchSpace.pptx` (research-journal repo, `roadmaps/gaussianArch/slides/`):
p.2 "3DGS Render: Workload", p.3 "3DGS Render: Architecture". Layout follows
`examples/` (`workloads/`, `arches/`, `mappings/`).

## Workload (p.2)

```
G2D[v, g]           = ComputeProject(G3D[g], View[v])
alpha[v, g, pu, pv] = ComputeAlpha(G2D[v, g], pu, pv)
I[v, pu, pv]        = ComputeBlend(F2D[v, g], alpha[v, g, pu, pv])      (reduce over g)

g in G(v)                    Gaussian index, near-to-far at view v
(pu, pv) in Np,row x Np,col  2D pixel index
v in [1, Nv]                 views to render
```

- Blend simplification 1: g is already depth-sorted per view, so a g reduction from
  small to large index resolves the depth dependency; the transmittance recurrence is
  carried in the per-pixel accumulator I (RGB + T).
- Blend assumption 1 (relaxed for now): g order is the same across views.
- **Dense association** (this first analysis): every Gaussian contributes to every pixel
  of every view, i.e. the contributed / intersected association matrices are all ones
  and the iteration space is the full box `Nv x Ng x Np,row x Np,col`.

| file | contents |
|---|---|
| `workloads/dense/gaussian_project/garden.yaml` | ComputeProject only (G2D written to DRAM) |
| `workloads/dense/alpha_compute/garden.yaml` | ComputeAlpha only (alpha written to DRAM) |
| `workloads/dense/alpha_blend/garden.yaml` | ComputeBlend only (alpha read from DRAM) |
| `workloads/dense/render/garden.yaml` | all three, as one cascade (G2D and alpha are intermediates the mapper may keep on chip) |

Garden ranks (MipNeRF360 garden, on-the-fly-nvs anchor): `Nv = 185` views,
`Ng = 1,041,334` Gaussians, `622 x 960` pixels, so 1.15e14 points per Einsum. Bits:
G3D 352 (mean3D + scale + quaternion + opacity, fp32), View 640 (world-to-camera 4x4 +
fx, fy, cx, cy), G2D 192 (mean2D + conic + opacity), F2D 96 (RGB), alpha 32, I 128 (RGB + T).
All are Jinja knobs (`Nv`, `Ng`, `NpRow`, `NpCol`, `AlphaBits`).

## Architecture (p.3)

`arches/3dgs_render.yaml`: External Memory (DRAM, LPDDR4) -> Global Buffer (CACTI SRAM,
`GlobalBufferBytes`, default 256 KiB) -> { ComputeProject unit array | ComputeAlpha unit
array | ComputeBlend unit array } (IntMAC, `ProjectUnits` / `AlphaUnits` / `BlendUnits` =
256 each; ~150 / 12 / 6 MAC-equivalents per projection / alpha / blend step). Memory constraints are the
`size` / `keep` / `may_keep` fields; compute constraints are the spatial fanouts and the
`enabled:` rule that pins each Einsum to its unit.

## Run

```bash
# from the accelforge repo root, env .conda/accelforge
python 3dgs_examples/run_dense_mapping.py                 # 4 workloads x 3 buffer sizes, ~15 s
python 3dgs_examples/run_dense_mapping.py --glb 131072 --workloads render --imperfect
```

Outputs: `mappings/dense/<workload>/garden/glb_<bytes>.{yaml,svg}` (the mapper's
LoopTree as tagged YAML and as a diagram) and `mappings/dense/summary_garden.json`.

## First results (dense garden, energy-optimal mappings)

| workload | GB | DRAM / frame-set | re-fetch (vs. read once) |
|---|---|---|---|
| gaussian_project | any | 4.67 GB | G3D, View, G2D x1 (all 185 View matrices held on chip) |
| alpha_compute | any | 460,100 GB | Alpha x1, G2D x1 (Alpha writes are 4 B x 1.15e14) |
| alpha_blend | 64 KiB / 256 KiB / 1 MiB | 460,500 / 460,200 / 460,200 GB | F2D x160 / x40 / x10 |
| render (fused) | 64 KiB / 256 KiB / 1 MiB | 2,163 / 284 / 76 GB | F2D x311 / x40 / x10; G2D spilled (1 write + 311 / 40 / 10 re-reads); Alpha never in DRAM |

The mapper fuses alpha -> blend but not projection: a top-level `Sequential` runs
ComputeProject over all (v, g) first and writes G2D to DRAM, and the alpha/blend nest
re-reads it per pixel strip. Keeping G2D on chip would need recomputing the projection
per strip or holding a whole image of I (9.6 MB). Per-tensor GLB occupancy and plots:
`notebooks/3dgs/dense/basic_arch_analysis.ipynb`.

Fused alpha -> blend nest at 256 KiB (`mappings/dense/render/garden/glb_262144.yaml`,
after the projection branch):

```
DRAM[G2D, I, F2D]
  for pv / 24                    40 pixel-column strips        <- irrelevant to G2D, F2D: x40 re-fetch
    for v / 1
      GB[I]                      I strip: 622 x 24 px x 16 B = 239 KB
      for g / 14
        GB[G2D, F2D]             14 Gaussians
        for pu / 2
          GB[Alpha]              alpha block, never leaves the chip
          Sequential: ComputeAlpha (alpha array) -> ComputeBlend (blend array)
```

The re-fetch factor is the trip count of the pixel loop above the Gaussian buffer (the
refetch rule in `plans/3dgs-accelforge/tutorial_workload_to_mapping.md`, Step 5b): a
bigger buffer holds a wider pixel strip of accumulators, so Gaussians are streamed fewer
times.

Caveats: dense association overcounts work by orders of magnitude (garden frame 0
blends 25.7M of 445M tile-local slots, and the dense box here is all-pixels x
all-Gaussians); latency and leak are distorted by hwcomponents' LPDDR4 throughput entry,
so compare DRAM traffic and dynamic energy, not frame time.

## Plan (next steps)

1. **Dense baseline** (this commit): mappings + LoopTrees for the three workloads.
2. **Buffer sweep**: add sizes to `--glb` and plot DRAM vs GB size per workload, as in
   `examples/3dgs/dram_vs_sram_sweep.py`.
3. **Association-aware workloads**: replace the dense box with the intersected /
   contributed association (per-tile lists, ragged sizes or a stencil approximation,
   then a sparsity pass; see `plans/3dgs-accelforge/discussion_1028.md`).
4. **Cross-frame reuse**: exploit `G2D[v, g]` similarity across views (blend
   assumption 1) once the per-view ordering is modeled.
