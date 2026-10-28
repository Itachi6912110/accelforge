"""Can AccelForge express the 3DGS alpha-compute / alpha-blend features?

Each probe writes a tiny workload YAML that tries to express one feature of renderCUDA,
loads it on examples/3dgs/arches/alpha_compute_array.yaml, runs the mapper, and records
whether AccelForge accepts it (and what it computes) or rejects it (and the error).

Usage (from the accelforge repo root, with the worktree's accelforge importable):
    python examples/3dgs/probe_limitations.py [--out DIR]
Writes DIR/probe_results.json (default: examples/3dgs/results/).
"""

import argparse
import json
import os
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import hwcomponents
import accelforge as af
from hwcomponents_extended.cache import SimpleSRAMCache

HERE = Path(__file__).resolve().parent
ARCH = HERE / "arches" / "alpha_compute_array.yaml"

PROBES = {
    "baseline_dense_tile": dict(
        question="Dense per-tile alpha compute (the formulation used by the sweep).",
        workload="""
            workload:
              iteration_space_shape: {q: 0 <= q < 256, k: 0 <= k < 744}
              einsums:
              - name: AlphaCompute
                tensor_accesses:
                - {name: GaussParams, projection: [k], bits_per_value: 192}
                - {name: Alpha, projection: [q, k], output: True, bits_per_value: 32}
            """,
        expect_computes=256 * 744,
    ),
    "sparsity_density": dict(
        question=(
            "Data-dependent sparsity: only ~6% of dense (pixel, Gaussian) slots blend "
            "(25.7M of 445M on garden frame 0) and ~65% are evaluated at all before "
            "early termination. Is there a density / sparsity annotation?"
        ),
        workload="""
            workload:
              iteration_space_shape: {q: 0 <= q < 256, k: 0 <= k < 744}
              einsums:
              - name: AlphaCompute
                tensor_accesses:
                - {name: GaussParams, projection: [k], bits_per_value: 192}
                - {name: Alpha, projection: [q, k], output: True, bits_per_value: 32,
                   density: 0.0578}
            """,
    ),
    "indirection_point_list": dict(
        question=(
            "Tile -> Gaussian indirection: renderCUDA reads "
            "GaussParams[point_list[range_start(t) + k]]. Can a projection index a "
            "tensor through another tensor's values?"
        ),
        workload="""
            workload:
              iteration_space_shape:
                t: 0 <= t < 4
                q: 0 <= q < 256
                k: 0 <= k < 744
              einsums:
              - name: AlphaCompute
                tensor_accesses:
                - {name: GaussParams, projection: {G: "PointList[t, k]"}, bits_per_value: 192}
                - {name: Alpha, projection: [t, q, k], output: True, bits_per_value: 32}
            """,
    ),
    "ragged_lists_isl_union": dict(
        question=(
            "Ragged per-tile list lengths (garden: p50 634, max 2710 Gaussians/tile). "
            "Data-independent per-tile lengths written as an ISL union over tiles."
        ),
        workload="""
            workload:
              einsums:
              - name: AlphaCompute
                iteration_space_shape:
                - 0 <= q < 256
                - 0 <= t < 4
                - 0 <= k
                - ((t = 0 and k < 600) or (t = 1 and k < 2700) or (t = 2 and k < 40) or (t = 3 and k < 634))
                tensor_accesses:
                - {name: GaussParams, projection: [t, k], bits_per_value: 192}
                - {name: Alpha, projection: [t, q, k], output: True, bits_per_value: 32}
            """,
        expect_computes=256 * (600 + 2700 + 40 + 634),
    ),
    "transmittance_recurrence": dict(
        question=(
            "Transmittance prefix product T[q, k+1] = T[q, k] * (1 - Alpha[q, k]): a "
            "tensor that is both read and written at a shifted index."
        ),
        workload="""
            workload:
              iteration_space_shape: {q: 0 <= q < 256, k: 0 <= k < 744}
              einsums:
              - name: Transmittance
                tensor_accesses:
                - {name: Alpha, projection: [q, k], bits_per_value: 32}
                - {name: T, projection: {Q: q, K: k}, bits_per_value: 32}
                - {name: T, projection: {Q: q, K: k + 1}, output: True, bits_per_value: 32}
            """,
    ),
    "cross_tile_reuse_affine_window": dict(
        question=(
            "Cross-tile Gaussian reuse (garden: a touched Gaussian overlaps 3.89 tiles "
            "on average). Only *affine* overlap is expressible: Gaussians binned by "
            "home tile b, tile t reads bins t + r, r in [0, R) -- a 1-D convolution "
            "window. Does the projection with an expression index work?"
        ),
        workload="""
            workload:
              iteration_space_shape:
                t: 0 <= t < 60
                r: 0 <= r < 4
                q: 0 <= q < 256
                j: 0 <= j < 186
              einsums:
              - name: AlphaCompute
                tensor_accesses:
                - {name: GaussParams, projection: {B: t + r, J: j}, bits_per_value: 192}
                - {name: Alpha, projection: [t, q, r, j], output: True, bits_per_value: 32}
            """,
        expect_computes=60 * 4 * 256 * 186,
    ),
}


def _unrolled_transmittance(n_steps):
    return dict(
        question=(
            "Workaround: unroll the recurrence into one Einsum per Gaussian step, "
            f"T{{k+1}}[q] = T{{k}}[q] * Alpha{{k}}[q], here {n_steps} steps (a garden "
            "tile needs ~743, i.e. one Einsum per tile-list slot)."
        ),
        workload="""
            workload:
              iteration_space_shape: {q: 0 <= q < 256}
              bits_per_value: {All: 32}
              einsums:
            """
        + "".join(
            f"""
              - name: T{k + 1}Step
                tensor_accesses:
                - {{name: T{k}, projection: [q]}}
                - {{name: Alpha{k}, projection: [q]}}
                - {{name: T{k + 1}, projection: [q], output: True}}
            """
            for k in range(n_steps)
        ),
        expect_computes=256 * n_steps,
    )


for _n in (8, 32, 128):
    PROBES[f"transmittance_unrolled_cascade_{_n}"] = _unrolled_transmittance(_n)


def _first_line(e: BaseException) -> str:
    lines = [l for l in str(e).strip().splitlines() if l.strip()]
    return (type(e).__name__ + ": " + (lines[0] if lines else "")).strip()[:400]


def run_probe(name, probe, tmp):
    f = Path(tmp) / f"{name}.yaml"
    f.write_text(textwrap.dedent(probe["workload"]))
    out = {"question": probe["question"], "workload": textwrap.dedent(probe["workload"])}
    try:
        spec = af.Spec.from_yaml(ARCH, f)
        spec.config.add_component_models(SimpleSRAMCache)
        spec.mapper.metrics = af.mapper.Metrics.ENERGY
        cwd = os.getcwd()
        os.chdir(tmp)  # the mapper writes a debug mapping.svg into the cwd
        t0 = time.perf_counter()
        try:
            m = spec.map_workload_to_arch()
        finally:
            os.chdir(cwd)
        out["mapper_seconds"] = round(time.perf_counter() - t0, 2)
        out["status"] = "accepted"
        out["n_computes"] = int(m.n_computes())
        if "expect_computes" in probe:
            out["expect_computes"] = probe["expect_computes"]
            out["computes_match"] = out["n_computes"] == probe["expect_computes"]
        out["dram_actions"] = {
            f"{t}.{a}": float(v)
            for (c, t, a), v in m.actions(per_component=True, per_tensor=True).items()
            if c == "DRAM" and v
        }
    except Exception as e:  # noqa: BLE001 - recording the rejection is the point
        out["status"] = "rejected"
        out["error"] = _first_line(e)
        out["error_detail"] = " ".join(str(e).split())[:1500]
        out["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
    return out


def energy_model_inventory():
    """Which hwcomponents models could cost an FP / exp-based alpha PE?"""
    names = sorted({m.__name__ for m in hwcomponents.get_models()})
    hits = [n for n in names if any(s in n.lower() for s in ("exp", "fp", "float", "sfu"))]
    arith = [n for n in names if n.startswith("Aladdin")]
    return {"n_models": len(names), "fp_or_exp_models": hits, "aladdin_arith": arith}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, probe in PROBES.items():
            results[name] = run_probe(name, probe, tmp)
    results["energy_models"] = energy_model_inventory()
    (args.out / "probe_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print()
    for name, r in results.items():
        if name == "energy_models":
            print(f"{name:34s} fp/exp models: {r['fp_or_exp_models']}  aladdin: {r['aladdin_arith']}")
            continue
        detail = r.get("error") or (
            f"n_computes={r['n_computes']}"
            + (f" match={r['computes_match']}" if "computes_match" in r else "")
        )
        print(f"{name:34s} {r['status']:9s} {detail}")


if __name__ == "__main__":
    main()
