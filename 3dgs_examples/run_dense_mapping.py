"""Map the dense 3DGS render workloads (3DGSArchSpace.pptx p.2) onto the p.3 arch.

Usage (from the accelforge repo root):
    python 3dgs_examples/run_dense_mapping.py [--scene garden] [--glb 65536 262144 1048576]
        [--workloads gaussian_project alpha_compute alpha_blend render] [--imperfect]

For every (workload, GlobalBuffer size) it runs the FFM mapper (energy objective) on
  3dgs_examples/workloads/dense/<workload>/<scene>.yaml  x  3dgs_examples/arches/3dgs_render.yaml
and writes
  3dgs_examples/mappings/dense/<workload>/<scene>/glb_<bytes>.yaml   LoopTree mapping
  3dgs_examples/mappings/dense/<workload>/<scene>/glb_<bytes>.svg    LoopTree diagram
  3dgs_examples/mappings/dense/summary_<scene>.json                  per-point numbers
"""

import argparse
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import accelforge as af
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_per_tensor_n_elements

HERE = Path(__file__).resolve().parent
ARCH = HERE / "arches" / "3dgs_render.yaml"
BITS_PER_ACTION = {"DRAM": 64, "GlobalBuffer": 256}

# LoopTree -> tagged YAML emitter shared with examples/3dgs (Mapping.to_yaml() cannot
# serialize mapper output; see that file).
_spec = importlib.util.spec_from_file_location(
    "dram_vs_sram_sweep", HERE.parent / "examples" / "3dgs" / "dram_vs_sram_sweep.py"
)
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)
mapping_to_yaml = _sweep.mapping_to_yaml

# One scratch cwd for the whole run: joblib's reused workers inherit the cwd of the
# first mapper call, so a per-call TemporaryDirectory leaves them in a deleted dir.
_SCRATCH = tempfile.TemporaryDirectory()


def map_point(workload: Path, glb_bytes: int, imperfect: bool):
    spec = af.Spec.from_yaml(ARCH, workload, jinja_parse_data={"GlobalBufferBytes": glb_bytes})
    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    if imperfect:  # garden's Ng = 2*7*74381 and 622 = 2*311 factor badly
        spec.mapper.explore_imperfect_temporal_loops = True
        spec.mapper.explore_imperfect_spatial_loops = True
    cwd = os.getcwd()
    os.chdir(_SCRATCH.name)  # the mapper writes a debug mapping.svg into the cwd
    try:
        t0 = time.perf_counter()
        m = spec.map_workload_to_arch()
        mapper_s = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    sizes = {}  # bits, per tensor, from every Einsum's evaluated spec
    for evaluated in m.evaluated_specs.values():
        n = get_per_tensor_n_elements(evaluated.workload)
        for e in evaluated.workload.einsums:
            for t in e.tensor_names:
                sizes[t] = n[t] * e.tensor_accesses[t].bits_per_value
    produced = {t for e in spec.workload.einsums for t in e.output_tensor_names}
    consumed = {t for e in spec.workload.einsums for t in e.input_tensor_names}
    intermediates = produced & consumed

    actions = m.actions(per_component=True, per_tensor=True)
    tensors = {}
    for t, bits in sizes.items():
        dram = sum(
            actions.get(("DRAM", t, a), 0) for a in ("read", "write")
        ) * BITS_PER_ACTION["DRAM"] / 8
        compulsory = 0.0 if t in intermediates else bits / 8
        tensors[t] = {
            "dram_bytes": dram,
            "compulsory_dram_bytes": compulsory,
            "refetch_factor": dram / compulsory if compulsory else None,
            # DRAM traffic in units of the tensor's size; for an intermediate that is
            # spilled (not fused) it is 1 write + its re-reads, 0 if it stays on chip
            "dram_over_size": dram / (bits / 8),
            "intermediate": t in intermediates,
        }
    # Peak GlobalBuffer occupancy per tensor: the mapper's
    # "<einsum><SEP>usage<SEP>memory<SEP>GlobalBuffer<SEP><tensor>" columns hold the
    # reserved fraction of the buffer. A tensor shared by two Einsums (fused Alpha)
    # appears once per Einsum with the same reservation, so take the max.
    glb_occupancy = {}
    for col in m.data.columns:
        parts = col.split("<SEP>")
        if len(parts) == 5 and parts[1:4] == ["usage", "memory", "GlobalBuffer"]:
            occ = float(m.data[col].iloc[0]) * glb_bytes
            glb_occupancy[parts[4]] = max(glb_occupancy.get(parts[4], 0.0), occ)
    energy = {k: float(v) for k, v in m.energy(per_component=True).items() if v}
    return m, {
        "glb_bytes": glb_bytes,
        "n_computes": int(m.n_computes()),
        "energy_J": float(m.energy()),
        "energy_per_component_J": energy,
        "latency_s": float(m.latency()),
        "dram_bytes": sum(v["dram_bytes"] for v in tensors.values()),
        "tensors": tensors,
        "glb_occupancy_bytes": glb_occupancy,
        "mapper_s": round(mapper_s, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden")
    ap.add_argument("--glb", type=int, nargs="+", default=[65536, 262144, 1048576])
    ap.add_argument("--workloads", nargs="+", default=["gaussian_project", "alpha_compute", "alpha_blend", "render"])
    ap.add_argument("--imperfect", action="store_true")
    args = ap.parse_args()

    out_root = HERE / "mappings" / "dense"
    summary = {}
    for wl in args.workloads:
        wfile = HERE / "workloads" / "dense" / wl / f"{args.scene}.yaml"
        odir = out_root / wl / args.scene
        odir.mkdir(parents=True, exist_ok=True)
        for glb in args.glb:
            m, row = map_point(wfile, glb, args.imperfect)
            summary.setdefault(wl, []).append(row)
            header = [
                f"AccelForge FFM mapper (metric: energy), dense {wl}, scene {args.scene}",
                f"arch: 3dgs_examples/arches/{ARCH.name} with GlobalBufferBytes={glb}",
                f"workload: 3dgs_examples/workloads/dense/{wl}/{args.scene}.yaml",
                f"energy {row['energy_J']:.6g} J, latency {row['latency_s']:.6g} s, "
                f"DRAM {row['dram_bytes'] / 1e9:.6g} GB",
            ]
            (odir / f"glb_{glb}.yaml").write_text(mapping_to_yaml(m.mapping(), header))
            (odir / f"glb_{glb}.svg").write_text(str(m.render()))
            print(
                f"{wl:14s} GLB {glb:>8d} B  E {row['energy_J']:.4g} J  "
                f"lat {row['latency_s']:.4g} s  DRAM {row['dram_bytes'] / 1e9:.4g} GB  "
                + "  ".join(
                    f"{t} x{v['refetch_factor']:.3g}" if v["refetch_factor"]
                    else f"{t} (via DRAM, x{v['dram_over_size']:.3g} size)" if v["dram_bytes"]
                    else f"{t} (on chip)"
                    for t, v in row["tensors"].items()
                )
                + "  GLB occ "
                + " ".join(f"{t} {b / glb:.1%}" for t, b in row["glb_occupancy_bytes"].items())
                + f"  [{row['mapper_s']} s]",
                flush=True,
            )
    (out_root / f"summary_{args.scene}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
