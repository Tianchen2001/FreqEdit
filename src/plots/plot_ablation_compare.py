"""Kontext vs Qwen band ablation, no QG on either side: A1 (D1+D2), A2 (D1), A3 (D2) minus native.
Rows = model x source, columns = metrics. Qwen A1 is its plain FreqEdit run.

    python plot_ablation_compare.py --table tables/nostyle_ablation_by_category_source.csv --turn 10
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_ablation import CATS, SOURCES, TEXT, MUTED, GRID, align_zero

METRICS = [("psnr", "PSNR (dB)", True), ("ssim", "SSIM", True), ("lpips", "LPIPS", False), ("dino", "DINO", True)]
MODELS = [("kontext", "Kontext", "freqedit_A1"), ("qwen", "Qwen", "freqedit")]
VARS = [("A1", "A1: D1+D2", "#eb6834"), ("freqedit_A2", "A2: D1 only", "#1baf7a"), ("freqedit_A3", "A3: D2 only", "#eda100")]

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True); ap.add_argument("--out", default="figs")
ap.add_argument("--name", default="ablation_kontext_vs_qwen"); ap.add_argument("--turn", type=int, default=10)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.table)
rows = [(mk, ml, a1, sk, sl) for mk, ml, a1 in MODELS for sk, sl in SOURCES]
fig, axes = plt.subplots(len(rows), len(METRICS), figsize=(2.5 * len(METRICS), 2.0 * len(rows) + 0.8),
                         constrained_layout=True, squeeze=False)
x = np.arange(len(CATS)); w = 0.8 / len(VARS)
for r, (mk, ml, a1, sk, sl) in enumerate(rows):
    d = df[df.model == mk].set_index(["method", "category", "source"])
    for c, (m, mlab, hi) in enumerate(METRICS):
        ax = axes[r, c]; ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.tick_params(length=0); ax.axhline(0, color=MUTED, linewidth=0.8)
        for j, (meth, lab, col) in enumerate(VARS):
            meth = a1 if meth == "A1" else meth
            vals = []
            for ck, _ in CATS:
                g = d.loc[(meth, ck, sk), f"{m}@{a.turn}"] - d.loc[("native", ck, sk), f"{m}@{a.turn}"]
                vals.append(g if hi else -g)
            ax.bar(x + (j - 1) * w, vals, w * 0.88, color=col, label=lab)
        ax.set_xticks(x, [cl for _, cl in CATS])
        if r == 0: ax.set_title(("Δ " if hi else "−Δ ") + mlab, color=TEXT)
        if c == 0: ax.set_ylabel(f"{ml}, {sl.split()[0].lower()}\nvariant − native")
align_zero(axes.ravel())
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3)
when = f"cumulative mean over turns 1–{a.turn}" if a.turn > 1 else "turn 1"
fig.suptitle(f"Band ablation without QG, Kontext vs Qwen, gain over native (↑ better), {when}", x=0.01, ha="left", fontsize=11)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"{a.name}_t{a.turn}.{ext}"), dpi=220, bbox_inches="tight")
print(f"{a.out}/{a.name}_t{a.turn}.png")
