"""Plot the dense-render mapping sweep written by run_dense_mapping.py.

Usage (from the accelforge repo root, after run_dense_mapping.py):
    python 3dgs_examples/plot_dense_mapping.py [--scene garden]

Reads 3dgs_examples/mappings/dense/summary_<scene>.json and writes
  3dgs_examples/mappings/dense/fig_dram_vs_glb_<scene>.png       DRAM traffic vs GLB size
  3dgs_examples/mappings/dense/fig_glb_occupancy_<scene>.png     GLB occupancy breakdown

"alpha compute + alpha blend" is the unfused pair run back to back: its DRAM traffic is
the sum of the two stages, and its GLB occupancy shows each tensor at its peak over the
two stages (G2D from alpha compute, F2D / I from alpha blend). The stages never hold the
buffer at the same time, so that bar is the union of both stages' tiles, an upper bound.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TENSORS = ["Alpha", "G2D", "F2D", "I"]
COLORS = {"Alpha": "#2a78d6", "G2D": "#eb6834", "F2D": "#1baf7a", "I": "#eda100"}
UNUSED = "#e1e0d9"
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
GROUPS = [
    ("alpha_compute", "alpha compute"),
    ("alpha_blend", "alpha blend"),
    ("pair", "alpha compute +\nalpha blend"),
    ("render", "render (fusion)"),
]
LINE_COLORS = {"alpha_compute": "#2a78d6", "alpha_blend": "#eb6834",
               "pair": "#1baf7a", "render": "#eda100"}


def kib(b):
    return f"{b // 1024} KiB" if b < 1 << 20 else f"{b // (1 << 20)} MiB"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(scene):
    summary = json.loads((HERE / "mappings" / "dense" / f"summary_{scene}.json").read_text())
    rows = {wl: {r["glb_bytes"]: r for r in rs} for wl, rs in summary.items()}
    glbs = sorted(rows["render"])
    dram = {wl: [rows[wl][g]["dram_bytes"] for g in glbs] for wl in rows}
    dram["pair"] = [a + b for a, b in zip(dram["alpha_compute"], dram["alpha_blend"])]
    occ = {wl: [rows[wl][g]["glb_occupancy_bytes"] for g in glbs] for wl in rows}
    occ["pair"] = [
        {t: max(c.get(t, 0.0), b.get(t, 0.0)) for t in TENSORS}
        for c, b in zip(occ["alpha_compute"], occ["alpha_blend"])
    ]
    return glbs, dram, occ


def plot_dram(scene, glbs, dram, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor=SURFACE)
    style(ax)
    for key, label in GROUPS:
        ax.plot(glbs, [d / 1e9 for d in dram[key]], marker="o", markersize=4,
                linewidth=1.6, color=LINE_COLORS[key], label=label.replace("\n", " "))
        # alpha compute and alpha blend land within 0.01% of each other: one shared label
        nudge = {"alpha_compute": -5, "alpha_blend": 5}.get(key, 0)
        ax.annotate(f"{dram[key][-1] / 1e9:,.0f} GB", (glbs[-1], dram[key][-1] / 1e9),
                    xytext=(6, nudge), textcoords="offset points", va="center",
                    fontsize=8, color=LINE_COLORS[key] if nudge else INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(glbs, [kib(g) for g in glbs])
    ax.set_xlim(glbs[0] / 1.3, glbs[-1] * 1.9)
    ax.set_xlabel("GlobalBuffer size", color=INK)
    ax.set_ylabel("DRAM traffic per run (GB, log)", color=INK)
    ax.set_title(f"Dense 3DGS render on {scene}: DRAM traffic vs GlobalBuffer size",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right",
              bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_occupancy(scene, glbs, occ, out):
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    style(ax)
    width, gap = 0.8, 1.0
    xs, xlabels, cells = [], [], {t: [] for t in TENSORS + ["total"]}
    for gi, (key, _) in enumerate(GROUPS):
        for bi, g in enumerate(glbs):
            x = gi * (len(glbs) + gap) + bi
            xs.append(x)
            xlabels.append(kib(g))
            bottom = 0.0
            for t in TENSORS:
                pct = 100 * occ[key][bi].get(t, 0.0) / g
                ax.bar(x, pct, width, bottom=bottom, color=COLORS[t],
                       edgecolor=SURFACE, linewidth=0.5,
                       label=t if (gi, bi) == (0, 0) else None)
                if pct >= 6:
                    ax.text(x, bottom + pct / 2, f"{pct:.1f}%", ha="center",
                            va="center", fontsize=7.5, color=INK)
                bottom += pct
                cells[t].append(f"{pct:.2f}" if pct else "-")
            ax.bar(x, max(0.0, 100 - bottom), width, bottom=bottom, color=UNUSED,
                   edgecolor=SURFACE, linewidth=0.5,
                   label="unused" if (gi, bi) == (0, 0) else None)
            cells["total"].append(f"{bottom:.2f}")
    for gi, (_, label) in enumerate(GROUPS):
        ax.text(gi * (len(glbs) + gap) + (len(glbs) - 1) / 2, 104, label, ha="center",
                va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs, xlabels, fontsize=8)
    ax.set_ylim(0, 116)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("GlobalBuffer capacity occupied (%)", color=INK)
    ax.set_title(f"Dense 3DGS render on {scene}: GlobalBuffer occupancy by tensor "
                 "(bars = GLB size)", color=INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=5, loc="upper left",
              bbox_to_anchor=(0, -0.08))
    table = ax.table(cellText=[cells[t] for t in TENSORS + ["total"]],
                     rowLabels=[f"{t} %" for t in TENSORS] + ["total %"],
                     colLabels=xlabels, cellLoc="center", bbox=[0, -0.62, 1, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK if r else INK2)
        cell.set_facecolor(SURFACE)
    fig.text(0.01, 0.01,
             "alpha compute + alpha blend: each tensor at its peak over the two sequential "
             "stages (they never share the buffer at once).\n"
             "Unfused Alpha streams DRAM <-> array and is never buffered.",
             fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.38)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden")
    args = ap.parse_args()
    glbs, dram, occ = load(args.scene)
    out_dir = HERE / "mappings" / "dense"
    plot_dram(args.scene, glbs, dram, out_dir / f"fig_dram_vs_glb_{args.scene}.png")
    plot_occupancy(args.scene, glbs, occ, out_dir / f"fig_glb_occupancy_{args.scene}.png")
    print("wrote", out_dir / f"fig_dram_vs_glb_{args.scene}.png")
    print("wrote", out_dir / f"fig_glb_occupancy_{args.scene}.png")


if __name__ == "__main__":
    main()
