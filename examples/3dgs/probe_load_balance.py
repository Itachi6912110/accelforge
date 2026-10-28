"""Does AccelForge see per-tile load imbalance?

Four tiles with ragged Gaussian-list lengths (600, 2700, 40, 634; written as an ISL union,
needs islpy-barvinok) run one-tile-per-engine on a 4-engine spatial array with unit-cost
actions (1 compute / cycle / engine, free memories), so latency is in cycles:

  load-aware latency      = max tile length            = 2700
  perfectly balanced      = total work / engines       = 3974 / 4 = 993.5
  uniform mean (no ragged)= ceil(mean length)          = 994

Two hand-written mappings are evaluated with evaluate_mapping -- one tile per engine
(spatial over tiles) and all tiles on one engine (temporal over tiles, exact latency =
total work = 3974) -- and the same workload is also handed to the mapper.

Usage (from the accelforge repo root): python examples/3dgs/probe_load_balance.py [--out DIR]
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import accelforge as af

HERE = Path(__file__).resolve().parent
LENGTHS = (600, 2700, 40, 634)

ARCH = """
arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0
    area: 0
    tensors: {keep: ~Intermediates, may_keep: All}
    actions:
    - {name: read, energy: 1, throughput: inf, latency: 0}
    - {name: write, energy: 1, throughput: inf, latency: 0}
  - !Container
    name: TileEngines
    spatial: [{name: T, fanout: 4}]
  - !Compute
    name: PE
    leak_power: 0
    area: 0
    actions:
    - {name: compute, energy: 1, throughput: 1, latency: 0}
"""


def workload(ragged: bool) -> str:
    if ragged:
        union = " or ".join(f"(t = {i} and k < {n})" for i, n in enumerate(LENGTHS))
        space = [f"0 <= t < {len(LENGTHS)}", "0 <= k", f"({union})"]
    else:
        mean = -(-sum(LENGTHS) // len(LENGTHS))
        space = [f"0 <= t < {len(LENGTHS)}", f"0 <= k < {mean}"]
    lines = [
        "workload:",
        "  bits_per_value: {All: 32}",
        "  einsums:",
        "  - name: AlphaCompute",
        "    iteration_space_shape:",
        *[f'    - "{s}"' for s in space],
        "    tensor_accesses:",
        "    - {name: GaussParams, projection: [t, k]}",
        "    - {name: Alpha, projection: [t, k], output: True}",
    ]
    return "\n".join(lines) + "\n"


# One tile per engine (spatial over tiles), or all tiles one after another on one engine.
MAPPINGS = {
    "spatial_tiles": """
mapping:
  nodes:
  - !Storage {tensors: [GaussParams, Alpha], component: MainMemory}
  - !Spatial {rank_variable: t, tile_shape: 1, name: T, component: TileEngines}
  - !Temporal {rank_variable: k, tile_shape: 1}
  - !Compute {einsum: AlphaCompute, component: PE}
""",
    "temporal_tiles": """
mapping:
  nodes:
  - !Storage {tensors: [GaussParams, Alpha], component: MainMemory}
  - !Temporal {rank_variable: t, tile_shape: 1}
  - !Temporal {rank_variable: k, tile_shape: 1}
  - !Compute {einsum: AlphaCompute, component: PE}
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "arch.yaml").write_text(ARCH)
        for mname, mtext in MAPPINGS.items():
            (t / f"{mname}.yaml").write_text(mtext)
        cwd = os.getcwd()
        os.chdir(tmp)  # the mapper writes a debug mapping.svg into the cwd
        try:
            for ragged in (True, False):
                name = "ragged" if ragged else "uniform_mean"
                (t / f"{name}.yaml").write_text(workload(ragged))
                hand = {}
                for mname in MAPPINGS:
                    ev = af.Spec.from_yaml(t / "arch.yaml", t / f"{name}.yaml", t / f"{mname}.yaml")
                    hand[mname] = ev.evaluate_mapping()
                r = hand["spatial_tiles"]
                spec = af.Spec.from_yaml(t / "arch.yaml", t / f"{name}.yaml")
                spec.mapper.metrics = af.mapper.Metrics.LATENCY
                m = spec.map_workload_to_arch()
                acts = m.actions(per_component=True, per_tensor=True)
                results[name] = {
                    "n_computes": int(r.n_computes()),
                    **{f"latency_{k}": float(v.latency()) for k, v in hand.items()},
                    "latency_mapper": float(m.latency()),
                    "mainmemory_reads_GaussParams": float(
                        acts.get(("MainMemory", "GaussParams", "read"), 0)
                    ),
                }
        finally:
            os.chdir(cwd)
    results["reference"] = {
        "spatial_tiles_load_aware_max": max(LENGTHS),
        "spatial_tiles_balanced": sum(LENGTHS) / len(LENGTHS),
        "temporal_tiles_exact": sum(LENGTHS),
        "exact_computes": sum(LENGTHS),
        "bounding_box_computes": max(LENGTHS) * len(LENGTHS),
    }
    (args.out / "probe_load_balance.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
