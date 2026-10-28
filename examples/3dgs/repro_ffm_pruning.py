"""Does FFM's tile-shape Pareto pruning lose the optimum? Matmul vs. sum-index workloads.

Two levels:

1. Formula level (`formula_level()`): call the partial-tile-shape criterion derivation,
   make_tile_shapes._make_evalable_objectives_from_formula, on two small formulas and
   show that the criterion it returns can rank two partial choices opposite to the true
   objective for some value of a not-yet-chosen tile shape:
     - a matmul-shaped refetch sum  64/(x*u) + 64/y          (x, y chosen, u not)
     - the stencil halo product     c*(1/a+1/b-1/(ab))*(1/c+1/d-1/(cd))  (a, c, d chosen)
2. Mapper level (`CASES`): each case is mapped twice in-process, as shipped and with the
   Pareto step inside make_tile_shapes.get_tile_shape_choices replaced by keep-all
   (exhaustive within each pmapping template). A lower energy with pruning disabled
   means the pruned search dropped the optimum.

Usage (from the accelforge repo root):
    python examples/3dgs/repro_ffm_pruning.py [--out DIR] [--only NAME ...]
Writes DIR/repro_ffm_pruning.json (default examples/3dgs/results/).
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy

import accelforge as af
import accelforge.mapper.FFM._make_pmappings.make_pmappings_from_templates.make_tile_shapes as mts
from hwcomponents_extended.cache import SimpleSRAMCache

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
SIMPLE = EXAMPLES / "arches" / "simple.yaml"  # MainMemory -> GlobalBuffer -> MAC
ARRAY = HERE / "arches" / "alpha_compute_array.yaml"  # DRAM -> GB -> 16x16 -> PE

WORKLOADS = {
    "matmul_mkn": """
        workload:
          iteration_space_shape: {{m: 0 <= m < {M}, k: 0 <= k < {K}, n: 0 <= n < {N}}}
          bits_per_value: {{All: 8}}
          einsums:
          - einsum: Z[m, n] = A[m, k] * B[k, n]
        """,
    "matmul": """
        workload:
          iteration_space_shape: {{m: 0 <= m < {N}, k: 0 <= k < {N}, n: 0 <= n < {N}}}
          bits_per_value: {{All: 8}}
          einsums:
          - einsum: Z[m, n] = A[m, k] * B[k, n]
        """,
    "conv1d": """
        workload:
          iteration_space_shape: {{p: 0 <= p < {N}, r: 0 <= r < 4}}
          bits_per_value: {{All: 8}}
          einsums:
          - name: Conv1D
            tensor_accesses:
            - {{name: I, projection: {{X: p + r}}}}
            - {{name: W, projection: [r]}}
            - {{name: O, projection: [p], output: True}}
        """,
    "conv2d": """
        workload:
          iteration_space_shape: {{p: 0 <= p < {N}, q: 0 <= q < {N}, r: 0 <= r < 3, s: 0 <= s < 3}}
          bits_per_value: {{All: 8}}
          einsums:
          - name: Conv2D
            tensor_accesses:
            - {{name: I, projection: {{Y: p + r, X: q + s}}}}
            - {{name: W, projection: [r, s]}}
            - {{name: O, projection: [p, q], output: True}}
        """,
    # The 3DGS stencil's structure with everything else removed: each output element
    # copies one input element through a 2x2 window, no weights, no reduction.
    "unfold2d": """
        workload:
          iteration_space_shape: {{ty: 0 <= ty < {N}, tx: 0 <= tx < {N}, ry: 0 <= ry < 2, rx: 0 <= rx < 2}}
          bits_per_value: {{All: 8}}
          einsums:
          - name: Unfold2D
            tensor_accesses:
            - {{name: G, projection: {{BY: ty + ry, BX: tx + rx}}}}
            - {{name: A, projection: [ty, tx, ry, rx], output: True}}
        """,
}

# (case name, workload (key of WORKLOADS, or a workload file), size N, arch, arch jinja)
CASES = [
    (f"{w}-N{n}-{an}-glb{g}", w, n, arch, jinja)
    for w, sizes in (
        ("matmul", (64, 128)),
        ("conv1d", (256,)),
        ("conv2d", (32,)),
        ("unfold2d", (16, 32)),
    )
    for n in sizes
    for an, arch, glbs, key in (
        ("simple", SIMPLE, (1024 * 8, 8192 * 8), "GlobalBufferSize"),
        ("array", ARRAY, (1024, 8192), "GlobalBufferBytes"),
    )
    for g in glbs
    for jinja in [{key: g, "MainMemoryEnergy": 100}]
] + [
    # Rectangular / non-power-of-two matmuls, three buffer sizes on both arches.
    (f"matmul-{m}x{k}x{n}-{an}-glb{g}", "matmul_mkn", (m, k, n), arch, {key: g, "MainMemoryEnergy": 100})
    for m, k, n in ((48, 96, 60), (120, 36, 90), (200, 64, 30), (96, 96, 96), (256, 16, 256),
                    (30, 200, 45), (60, 60, 240), (1024, 64, 64))
    for an, arch, glbs, key in (
        ("simple", SIMPLE, (512 * 8, 4096 * 8, 32768 * 8), "GlobalBufferSize"),
        ("array", ARRAY, (512, 4096, 32768), "GlobalBufferBytes"),
    )
    for g in glbs
] + [
    # The 3DGS stencil alpha compute, shrunk: one Gaussian per bin, small tile grids.
    (
        f"stencil-{tr}x{tc}-q{q}-array{ad}",
        HERE / "workloads" / "alpha_compute_stencil.yaml",
        None,
        ARRAY,
        dict(TileRows=tr, TileCols=tc, PixelsPerTile=q, ArrayDim=ad, GaussiansPerBin=1,
             GlobalBufferBytes=8 * 1024 * 1024),
    )
    for tr, tc, q, ad in ((8, 8, 64, 8), (16, 16, 16, 4), (8, 8, 16, 4))
]


def formula_level():
    """Criterion derivation vs. the true objective on two tiny formulas."""
    derive = mts._make_evalable_objectives_from_formula
    out = []
    x, y, u = sympy.symbols("stride0 stride1 stride2", positive=True, integer=True)
    a, b, c, d = sympy.symbols("stride3 stride4 stride5 stride6", positive=True, integer=True)
    halo = lambda p, r: 1 / p + 1 / r - 1 / (p * r)
    for name, f, known, bounds, A, B, unknown, values in (
        ("matmul-shaped refetch sum", 64 / (x * u) + 64 / y, {x, y},
         ((x, 1, 8), (y, 1, 8), (u, 1, 8)), {x: 1, y: 4}, {x: 4, y: 2}, u, (1, 8)),
        ("stencil halo product", sympy.expand(halo(a, b) * halo(c, d)), {a, c, d},
         ((a, 1, 2), (b, 1, 60), (c, 1, 2), (d, 1, 39)), {a: 2, c: 2, d: 39},
         {a: 1, c: 2, d: 39}, b, (1, 60)),
    ):
        goals, _ = derive(f, frozenset(known), bounds, 0, 0)
        row = {
            "name": name,
            "formula": str(f),
            "known": sorted(map(str, known)),
            "criteria": {str(k): v.goal for k, v in goals.items()},
            "criterion_A": {str(k): float(k.subs(A)) for k in goals},
            "criterion_B": {str(k): float(k.subs(B)) for k in goals},
            "true": {
                f"{unknown}={v}": {"A": float(f.subs({**A, unknown: v})),
                                   "B": float(f.subs({**B, unknown: v}))}
                for v in values
            },
        }
        out.append(row)
        print(f"[formula] {name}: criteria {row['criteria']}")
        print(f"          criterion A {row['criterion_A']}  B {row['criterion_B']}")
        for k, v in row["true"].items():
            print(f"          true objective at {k}: A {v['A']:.6g}  B {v['B']:.6g}")
    return out


def map_once(arch, workload_file, jinja):
    spec = af.Spec.from_yaml(arch, workload_file, jinja_parse_data=jinja)
    spec.config.add_component_models(SimpleSRAMCache)
    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the mapper writes a debug mapping.svg into the cwd
        try:
            t0 = time.perf_counter()
            m = spec.map_workload_to_arch()
            return float(m.energy()), time.perf_counter() - t0
        finally:
            os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    af.set_n_parallel_jobs(1, print_message=False)  # the patch must reach the mapper

    shipped = mts.makepareto_numpy
    keep_all = lambda mappings, goals, **kw: np.ones(mappings.shape[0], dtype=bool)

    formulas = formula_level()
    results = []
    with tempfile.TemporaryDirectory() as wdir:
        for name, wl, n, arch, jinja in CASES:
            if args.only and not any(o in name for o in args.only):
                continue
            if isinstance(wl, Path):
                wf = wl
            else:
                wf = Path(wdir) / f"{name}.yaml"
                dims = dict(zip("MKN", n)) if isinstance(n, tuple) else {"N": n}
                wf.write_text(WORKLOADS[wl].format(**dims))
            row = {"case": name, "workload": str(getattr(wl, "name", wl)), "N": n,
                   "arch": arch.name, **jinja}
            for label, fn in (("pruned", shipped), ("exhaustive", keep_all)):
                mts.makepareto_numpy = fn
                try:
                    row[f"{label}_J"], row[f"{label}_s"] = map_once(arch, wf, jinja)
                finally:
                    mts.makepareto_numpy = shipped
            row["loss_pct"] = 100 * (row["pruned_J"] / row["exhaustive_J"] - 1)
            results.append(row)
            flag = "LOST OPTIMUM" if row["loss_pct"] > 1e-6 else "ok"
            print(
                f"{name:32s} pruned {row['pruned_J']:.6g} J [{row['pruned_s']:.1f}s]  "
                f"exhaustive {row['exhaustive_J']:.6g} J [{row['exhaustive_s']:.1f}s]  "
                f"{row['loss_pct']:+.3f}%  {flag}",
                flush=True,
            )
    (args.out / "repro_ffm_pruning.json").write_text(
        json.dumps({"formula_level": formulas, "mapper_level": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
