"""Per-turn tone and fine-scale drift, VAE-only chain vs identity chains (native, FreqEdit), per model.
    python plots/plot_color_drift.py --table tables/color_drift.csv --out figs  (from src/)
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_ablation import TEXT, MUTED, GRID

ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--out", default="figs")
ap.add_argument("--set", default="output_identity", help="chain set to compare with the VAE chain")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.table)
df["model"] = df.run.str.split("_").str[0]; df["method"] = df.run.str.split("_", n=1).str[1]
METRICS = [("d_L_p01", "Black point ΔL* (p1)\n↑ = lifted blacks"), ("d_L_std", "Contrast Δstd(L*)\n↓ = flatter"),
           ("d_C_mean", "Chroma ΔC*\n↓ = desaturated"), ("hf_ratio", "Fine-scale energy / turn 0\n>1 grain, <1 smoothing")]
LINES = [("output_vae", "vae", "VAE round trip only", "#52514e", "--"),
         (a.set, "native", "Native", "#eb6834", "-"), (a.set, "freqedit", "FreqEdit", "#2a78d6", "-")]
MODELS = [("kontext", "FLUX.1 Kontext"), ("qwen", "Qwen-Image-Edit")]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), constrained_layout=True, sharex=True)
for r, (mk, ml) in enumerate(MODELS):
    for c, (m, lab) in enumerate(METRICS):
        ax = axes[r, c]; ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.axhline(1 if m == "hf_ratio" else 0, color=MUTED, lw=0.8)
        for folder, meth, name, col, ls in LINES:
            d = df[(df.folder == folder) & (df.model == mk) & (df.method == meth)]
            g = d.groupby("turn")[m]
            ctr = g.median() if m == "hf_ratio" else g.mean()
            ax.plot(ctr.index, ctr.values, ls, color=col, lw=2, marker="o", ms=3.5, label=f"{name} (n={d.image.nunique()})")
        if m == "hf_ratio": ax.set_yscale("log")
        if r == 0: ax.set_title(lab, fontsize=9, color=TEXT)
        if c == 0: ax.set_ylabel(ml, fontsize=10, color=TEXT)
        if r == 1: ax.set_xlabel("turn", color=MUTED)
for c in range(3):  # share y within a metric column across models
    lo = min(axes[r, c].get_ylim()[0] for r in range(2)); hi = max(axes[r, c].get_ylim()[1] for r in range(2))
    for r in range(2): axes[r, c].set_ylim(lo, hi)
lo = min(axes[r, 3].get_ylim()[0] for r in range(2)); hi = max(axes[r, 3].get_ylim()[1] for r in range(2))
for r in range(2): axes[r, 3].set_ylim(lo, hi)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, [x.split(" (n=")[0] for x in l], loc="outside lower center", ncol=3, frameon=False)
name = "identity chains" if "identity" in a.set else "editing chains"
fig.suptitle(f"Tone and fine-scale drift per turn, relative to turn 0: VAE-only chain vs {name} (mean; fine-scale: median)",
             x=0.01, ha="left", fontsize=11, color=TEXT)
tag = a.set.replace("output_", "")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"color_drift_vae_vs_{tag}.{ext}"), dpi=220, bbox_inches="tight")
print("ok")
