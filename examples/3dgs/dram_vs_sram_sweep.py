"""DRAM traffic vs. on-chip GlobalBuffer size for 3DGS alpha compute / blend.

For every (variant, stage, GlobalBuffer size) point this runs AccelForge's mapper with an
energy objective on

  stage alpha_compute: workloads/alpha_compute[_stencil].yaml on arches/alpha_compute_array.yaml
  stage alpha_blend:   workloads/alpha_blend[_stencil].yaml   on arches/alpha_blend_array.yaml
  stage fused:         workloads/alpha_fused.yaml             on arches/alpha_fused_arrays.yaml
                       (both Einsums, Alpha an intermediate the mapper may keep on chip)

and records the best-found mapping's DRAM / GlobalBuffer traffic per tensor, the
compulsory (read-once / write-once; 0 for intermediates) DRAM traffic, and the energy
breakdown. Variants (a (variant, stage) pair without a workload file is skipped):

  tile     per-tile pre-gathered Gaussian lists, one Einsum instance per tile
           (n_instances), i.e. no cross-tile reuse visible to the mapper
  stencil  tiles as explicit ranks and Gaussians binned so each bin is read by a 2x2
           window of tiles -- cross-tile reuse as an affine stencil

Usage (from the accelforge repo root):
    python examples/3dgs/dram_vs_sram_sweep.py [--out DIR] [--sizes B ...]
        [--variants tile stencil] [--stages alpha_compute alpha_blend fused] [--plot-only]

Writes into DIR (default examples/3dgs/results/); rows of the (variant, stage) pairs that
are re-run replace the old ones, the rest of the CSVs is kept:
  sweep_tensors.csv   one row per (variant, stage, size, tensor): DRAM / GlobalBuffer bytes
  sweep_summary.csv   one row per (variant, stage, size): totals, energy, latency
  mappings/<variant>/<stage>/glb_<bytes>.{yaml,svg}   the LoopTree for that point
  fig_dram_total.png, fig_dram_per_tensor.png, fig_energy.png, fig_fused_vs_unfused.png
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import accelforge as af
from accelforge.frontend.mapping.mapping import MappingNodeWithChildren
from accelforge.mapper.FFM._make_pmappings.make_pmappings import get_per_tensor_n_elements
from hwcomponents_extended.cache import SimpleSRAMCache

HERE = Path(__file__).resolve().parent

# stage -> (arch file stem, workload file stem)
STAGES = {
    "alpha_compute": ("alpha_compute_array", "alpha_compute"),
    "alpha_blend": ("alpha_blend_array", "alpha_blend"),
    "fused": ("alpha_fused_arrays", "alpha_fused"),
}
VARIANTS = {"tile": "{workload}.yaml", "stencil": "{workload}_stencil.yaml"}
# 128 B (smallest 4-way x 32 B SimpleSRAMCache) .. 8 MiB
DEFAULT_SIZES = [2**i for i in range(7, 24)]

# bits_per_action of the read/write actions in arches/*.yaml
BITS_PER_ACTION = {"DRAM": 64, "GlobalBuffer": 256}


# --------------------------------------------------------------------------- mapping
def _flow(v):
    if isinstance(v, str) and v.replace("_", "").isalnum() and not v[0].isdigit():
        return v
    if isinstance(v, list):
        return "[" + ", ".join(_flow(x) for x in v) + "]"
    return json.dumps(v)


def _emit(node, pad, lines):
    # component_object is the resolved arch node; it is left out because
    # Mapping.to_yaml() cannot serialize mapper output (InvertibleSet.__deepcopy__
    # drops the private _bits_per_value its serializer reads).
    fields = node.model_dump(exclude={"component_object", "nodes"}, exclude_defaults=True)
    lines.append(f"{pad}- !{type(node).__name__}")
    for k, v in fields.items():
        if v is not None:
            lines.append(f"{pad}  {k}: {_flow(v)}")
    if isinstance(node, MappingNodeWithChildren):
        lines.append(f"{pad}  nodes:")
        for child in node.nodes:
            _emit(child, pad + "  ", lines)


def mapping_to_yaml(mapping, header):
    """LoopTree YAML in the tagged format of examples/mappings/*.yaml."""
    lines = [f"# {line}" for line in header] + ["mapping:", "  nodes:"]
    for node in mapping.nodes:
        _emit(node, "  ", lines)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- sweep
def workload_file(variant, stage):
    return HERE / "workloads" / VARIANTS[variant].format(workload=STAGES[stage][1])


def run_point(variant, stage, glb_bytes, out):
    arch = HERE / "arches" / f"{STAGES[stage][0]}.yaml"
    workload = workload_file(variant, stage)
    spec = af.Spec.from_yaml(
        arch, workload, jinja_parse_data={"GlobalBufferBytes": glb_bytes}
    )
    spec.config.add_component_models(SimpleSRAMCache)
    spec.mapper.metrics = af.mapper.Metrics.ENERGY

    cwd = os.getcwd()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # the mapper writes a debug mapping.svg into the cwd
        try:
            m = spec.map_workload_to_arch()
        finally:
            os.chdir(cwd)
    mapper_s = time.perf_counter() - t0

    n_instances = spec.workload.n_instances
    # Per Einsum instance. Mappings.per_tensor_size() only looks at the first Einsum's
    # evaluated spec, so collect the sizes from every Einsum's.
    tensor_bits = {}
    for evaluated in m.evaluated_specs.values():
        n_elements = get_per_tensor_n_elements(evaluated.workload)
        for einsum in evaluated.workload.einsums:
            for tensor in einsum.tensor_names:
                tensor_bits[tensor] = (
                    n_elements[tensor] * einsum.tensor_accesses[tensor].bits_per_value
                )
    produced = {t for e in spec.workload.einsums for t in e.output_tensor_names}
    consumed = {t for e in spec.workload.einsums for t in e.input_tensor_names}
    intermediates = produced & consumed  # need not touch DRAM at all
    outputs = produced - intermediates
    actions = m.actions(per_component=True, per_tensor=True)

    def nbytes(component, tensor, action):
        return actions.get((component, tensor, action), 0.0) * BITS_PER_ACTION[component] / 8

    tensor_rows = []
    for tensor, bits in tensor_bits.items():
        row = {
            "variant": variant,
            "stage": stage,
            "glb_bytes": glb_bytes,
            "tensor": tensor,
            "is_output": tensor in outputs,
            "is_intermediate": tensor in intermediates,
            "compulsory_dram_bytes": 0.0 if tensor in intermediates else bits * n_instances / 8,
            "dram_read_bytes": nbytes("DRAM", tensor, "read"),
            "dram_write_bytes": nbytes("DRAM", tensor, "write"),
            "glb_read_bytes": nbytes("GlobalBuffer", tensor, "read"),
            "glb_write_bytes": nbytes("GlobalBuffer", tensor, "write"),
        }
        row["dram_bytes"] = row["dram_read_bytes"] + row["dram_write_bytes"]
        # Sanity: an optimal mapping can never beat read-once / write-once.
        assert row["dram_bytes"] >= row["compulsory_dram_bytes"] * (1 - 1e-9), row
        tensor_rows.append(row)

    # Leak is split out: it is leak_power x runtime, and the stock hwcomponents_cacti
    # LPDDR4 throughput (50 * 1024**2 accesses/s, ~0.42 GB/s at 64 bit) makes the
    # DRAM-bound runtime ~two orders of magnitude too long, so leak is inflated with it.
    energy = {"dyn": {}, "leak": {}}
    for (component, action), joules in m.energy(per_component=True, per_action=True).items():
        kind = "leak" if action == "leak" else "dyn"
        energy[kind][component] = energy[kind].get(component, 0.0) + float(joules)
    latency = m.latency(per_component=True)
    summary = {
        "variant": variant,
        "stage": stage,
        "glb_bytes": glb_bytes,
        "n_computes": int(m.n_computes()) * n_instances,
        "dram_bytes": sum(r["dram_bytes"] for r in tensor_rows),
        "compulsory_dram_bytes": sum(r["compulsory_dram_bytes"] for r in tensor_rows),
        "glb_bytes_moved": sum(r["glb_read_bytes"] + r["glb_write_bytes"] for r in tensor_rows),
        "energy_total_J": float(m.energy()),
        "latency_s": float(m.latency()),
        "mapper_s": round(mapper_s, 2),
    }
    summary["energy_dyn_total_J"] = sum(energy["dyn"].values())
    summary["energy_leak_total_J"] = sum(energy["leak"].values())
    for kind in ("dyn", "leak"):
        for component, joules in energy[kind].items():
            summary[f"energy_{kind}_{component}_J"] = joules
    for component, seconds in latency.items():
        summary[f"latency_{component}_s"] = float(seconds)

    mdir = out / "mappings" / variant / stage
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"glb_{glb_bytes}.yaml").write_text(
        mapping_to_yaml(
            m.mapping(),
            [
                f"AccelForge mapper (metric: energy), variant={variant} stage={stage}",
                f"arch: examples/3dgs/arches/{arch.name} with GlobalBufferBytes={glb_bytes}",
                f"workload: examples/3dgs/workloads/{workload.name}",
                f"DRAM bytes {summary['dram_bytes']:.6g} (compulsory "
                f"{summary['compulsory_dram_bytes']:.6g}), energy "
                f"{summary['energy_total_J']:.6g} J",
            ],
        )
    )
    # LoopTree diagram (graphviz): storage nodes, loops with tile shapes, reservations.
    (mdir / f"glb_{glb_bytes}.svg").write_text(str(m.render()))
    return tensor_rows, summary


def sweep(args):
    import pandas as pd

    # Keep rows of (variant, stage) pairs that are not re-run in this invocation.
    pairs = {(v, s) for v in args.variants for s in args.stages if workload_file(v, s).exists()}
    old = {}
    for name in ("sweep_tensors.csv", "sweep_summary.csv"):
        path = args.out / name
        if path.exists():
            df = pd.read_csv(path)
            old[name] = df[[(v, s) not in pairs for v, s in zip(df.variant, df.stage)]]
        else:
            old[name] = pd.DataFrame()

    tensor_rows, summaries = [], []
    for variant in args.variants:
        for stage in args.stages:
            if (variant, stage) not in pairs:
                print(f"skip {variant} {stage}: no {workload_file(variant, stage).name}")
                continue
            for glb_bytes in args.sizes:
                rows, summary = run_point(variant, stage, glb_bytes, args.out)
                tensor_rows += rows
                summaries.append(summary)
                print(
                    f"{variant:8s} {stage:14s} GLB {glb_bytes:>9d} B  "
                    f"DRAM {summary['dram_bytes'] / 1e6:12.3f} MB "
                    f"(compulsory {summary['compulsory_dram_bytes'] / 1e6:12.3f} MB)  "
                    f"E {summary['energy_total_J']:.4g} J  [{summary['mapper_s']} s]",
                    flush=True,
                )
                # Rewrite after every point so a long sweep can be inspected / resumed.
                for name, rows in (("sweep_tensors.csv", tensor_rows),
                                   ("sweep_summary.csv", summaries)):
                    pd.concat([old[name], pd.DataFrame(rows)], ignore_index=True).to_csv(
                        args.out / name, index=False
                    )


# --------------------------------------------------------------------------- plots
# dataviz reference palette (light), categorical slots 1-3 validated all-pairs.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
VARIANT_LABEL = {
    "tile": "per-tile lists (no cross-tile reuse)",
    "stencil": "2x2 stencil (cross-tile reuse)",
}
STAGE_LABEL = {
    "alpha_compute": "Alpha compute",
    "alpha_blend": "Alpha blend",
    "fused": "Fused alpha compute -> blend",
}
UNFUSED = ("alpha_compute", "alpha_blend")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xscale("log", base=2)
    ticks = [2**i for i in range(7, 24, 4)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_bytes(t) for t in ticks])
    ax.minorticks_off()


def _fmt_bytes(b):
    for unit, scale in (("MiB", 2**20), ("KiB", 2**10)):
        if b >= scale:
            return f"{b / scale:g} {unit}"
    return f"{b:g} B"


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from matplotlib.ticker import FuncFormatter, LogLocator

    # Components with no accesses at a point (e.g. a bypassed GlobalBuffer) are absent.
    summary = pd.read_csv(out / "sweep_summary.csv")
    summary[[c for c in summary.columns if c.startswith("energy_")]] = summary[
        [c for c in summary.columns if c.startswith("energy_")]
    ].fillna(0.0)
    tensors = pd.read_csv(out / "sweep_tensors.csv")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})
    variants = [v for v in VARIANTS if v in set(summary.variant)]
    stages = [s for s in UNFUSED if s in set(summary.stage)]  # figures 1-3

    # 1. Total DRAM traffic vs GlobalBuffer size, one panel per stage.
    fig, axes = plt.subplots(1, len(stages), figsize=(4.6 * len(stages), 3.4), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        _style(ax)
        for color, variant in zip(SERIES, variants):
            d = summary[(summary.stage == stage) & (summary.variant == variant)]
            ax.plot(d.glb_bytes, d.dram_bytes / 1e9, color=color, lw=2, marker="o",
                    ms=4, label=VARIANT_LABEL[variant])
        ax.set_ylim(0, summary[summary.stage == stage].dram_bytes.max() / 1e9 * 1.1)
        ax.set_title(STAGE_LABEL[stage], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("GlobalBuffer size", color=INK2)
        ax.set_ylabel("DRAM traffic per frame (GB)", color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("Unfused DRAM traffic: the compulsory Alpha round trip dominates at every size",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_dram_total.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. The Gaussian-side input tensors, where SRAM size can matter (log y). Alpha is
    #    always compulsory and PixelOut is always written exactly once, so they are
    #    left to the tables.
    panels = [(s, t) for s in stages for t in ("GaussParams", "Color")
              if ((tensors.stage == s) & (tensors.tensor == t)).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False)
    for ax, (stage, tensor) in zip(axes[0], panels):
        _style(ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        sel = tensors[(tensors.stage == stage) & (tensors.tensor == tensor)]
        ax.set_ylim(sel.compulsory_dram_bytes.min() / 1e6 / 1.6, sel.dram_bytes.max() / 1e6 * 1.6)
        for color, variant in zip(SERIES, variants):
            d = sel[sel.variant == variant]
            ax.plot(d.glb_bytes, d.dram_bytes / 1e6, color=color, lw=2, marker="o", ms=4,
                    label=VARIANT_LABEL[variant])
            ax.axhline(d.compulsory_dram_bytes.iloc[0] / 1e6, color=color, lw=1,
                       ls=(0, (3, 3)), alpha=0.7)
        ax.set_title(f"{STAGE_LABEL[stage]}: {tensor}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("GlobalBuffer size", color=INK2)
        ax.set_ylabel("DRAM traffic per frame (MB)", color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Gaussian-side inputs (dashed = compulsory, read once)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_dram_per_tensor.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3. Dynamic energy breakdown vs GlobalBuffer size (per-tile variant), stacked areas.
    #    Leak is excluded: it scales with the stock LPDDR4 model's far-too-long runtime.
    fig, axes = plt.subplots(1, len(stages), figsize=(4.6 * len(stages), 3.4), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        _style(ax)
        d = summary[(summary.stage == stage) & (summary.variant == variants[0])]
        pe = [c for c in d.columns if c.startswith("energy_dyn_") and c.endswith("PE_J")
              and d[c].sum() > 0][0]
        parts = [("DRAM", "energy_dyn_DRAM_J"), ("GlobalBuffer", "energy_dyn_GlobalBuffer_J"),
                 (pe[len("energy_dyn_"):-2], pe)]
        ax.stackplot(d.glb_bytes, *[d[c] * 1e3 for _, c in parts], colors=SERIES,
                     labels=[n for n, _ in parts], edgecolor=SURFACE, linewidth=1)
        ax.set_ylim(0, d.energy_dyn_total_J.max() * 1e3 * 1.25)
        ax.set_title(f"{STAGE_LABEL[stage]}, per-tile lists", color=INK, fontsize=10,
                     loc="left", pad=22)
        ax.set_xlabel("GlobalBuffer size", color=INK2)
        ax.set_ylabel("Dynamic energy per frame (mJ)", color=INK2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3, loc="lower left",
                  bbox_to_anchor=(0, 1.0), handlelength=1.2, handletextpad=0.5,
                  borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_energy.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 4. Fused vs unfused (per-tile variant): total DRAM traffic, and the fused design's
    #    dynamic energy breakdown.
    tile = summary[summary.variant == "tile"]
    fused = tile[tile.stage == "fused"].sort_values("glb_bytes")
    if fused.empty or not set(UNFUSED) <= set(tile.stage):
        return
    unfused = (
        tile[tile.stage.isin(UNFUSED)]
        .groupby("glb_bytes")[["dram_bytes", "compulsory_dram_bytes"]]
        .sum()
        .reset_index()
    )
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    _style(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    for color, (label, d) in zip(
        SERIES, (("unfused (compute + blend)", unfused), ("fused", fused))
    ):
        ax.plot(d.glb_bytes, d.dram_bytes / 1e6, color=color, lw=2, marker="o", ms=4,
                label=label)
        ax.axhline(d.compulsory_dram_bytes.iloc[0] / 1e6, color=color, lw=1,
                   ls=(0, (3, 3)), alpha=0.7)
    ax.set_title("DRAM traffic (dashed = compulsory)", color=INK, fontsize=10, loc="left")
    ax.set_xlabel("GlobalBuffer size", color=INK2)
    ax.set_ylabel("DRAM traffic per frame (MB)", color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")

    _style(ax2)
    pes = [c for c in fused.columns if c.startswith("energy_dyn_") and c.endswith("PE_J")]
    parts = [fused.energy_dyn_DRAM_J, fused.energy_dyn_GlobalBuffer_J, fused[pes].sum(axis=1)]
    ax2.stackplot(fused.glb_bytes, *[p * 1e3 for p in parts], colors=SERIES,
                  labels=["DRAM", "GlobalBuffer", "PE arrays"], edgecolor=SURFACE,
                  linewidth=1)
    ax2.set_ylim(0, fused.energy_dyn_total_J.max() * 1e3 * 1.25)
    ax2.set_title("Fused: dynamic energy", color=INK, fontsize=10, loc="left", pad=22)
    ax2.set_xlabel("GlobalBuffer size", color=INK2)
    ax2.set_ylabel("Dynamic energy per frame (mJ)", color=INK2)
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3, loc="lower left",
               bbox_to_anchor=(0, 1.0), handlelength=1.2, handletextpad=0.5,
               borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_fused_vs_unfused.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        sweep(args)
    plot(args.out)


if __name__ == "__main__":
    main()
