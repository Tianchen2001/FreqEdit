"""Band ablation (A group) vs native: one bar per variant, rows = image source, columns = metrics.

    python summarize.py output_identity output_identity_A1 output_identity_A2 output_identity_A3 output_identity_A9 \
        --out tables/identity_ablation
    python plot_ablation.py --table tables/identity_ablation_by_category_source.csv --out figs --name identity_ablation

Gain = variant - native (Kontext), LPIPS sign flipped so up = better; cumulative mean over turns 1..10.
Every panel shares the height of y = 0.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [("psnr", "PSNR (dB)", True), ("ssim", "SSIM", True), ("lpips", "LPIPS", False),
           ("clip_i", "CLIP-I", True), ("dino", "DINO", True)]
CATS = [("textured", "Textured"), ("smooth", "Smooth"), ("human", "Human")]
SOURCES = [("real", "Real images"), ("gen", "Generated images")]
# method name in summarize.py output -> label, color (reference categorical palette, fixed order)
VARIANTS = [
    ("freqedit", "FreqEdit (D1+D2, with QG)", "#2a78d6"),
    ("freqedit_A1", "A1: D1+D2, no QG", "#eb6834"),
    ("freqedit_A2", "A2: D1 only", "#1baf7a"),
    ("freqedit_A3", "A3: D2 only", "#eda100"),
    ("freqedit_A9", "A9: LL+D1+D2", "#e87ba4"),
]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 10, "text.color": TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def align_zero(axes, pad=0.08):
    spans = [(max(-lo, 0.0), max(hi, 0.0)) for lo, hi in (ax.get_ylim() for ax in axes)]
    frac = min(max([n / (n + p) for n, p in spans if n + p > 0] + [0.0]), 0.9)
    for ax, (neg, pos) in zip(axes, spans):
        total = max(neg / frac if frac > 0 else 0, pos / (1 - frac)) * (1 + pad)
        ax.set_ylim(-frac * total, (1 - frac) * total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="*_by_category_source.csv from summarize.py")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--name", default="ablation")
    ap.add_argument("--model", default="kontext")
    ap.add_argument("--turn", type=int, default=10)
    ap.add_argument("--title", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="method names to leave out, e.g. freqedit_A9")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.table)
    df = df[df.model == args.model]
    keys = ["category", "source"]
    nat = df[df.method == "native"].set_index(keys)
    variants = [v for v in VARIANTS if (df.method == v[0]).any() and v[0] not in args.exclude]
    if args.model == "qwen":  # Qwen has no QG: its FreqEdit run is A1
        variants = [("freqedit", "FreqEdit = A1: D1+D2 (no QG on Qwen)", VARIANTS[1][2]) if v[0] == "freqedit" else v
                    for v in variants]
    missing = [v[0] for v in VARIANTS if v[0] not in [u[0] for u in variants] and v[0] not in args.exclude
               and not (args.model == "qwen" and v[0] in ("freqedit_A1", "freqedit_A9"))]
    if missing:
        print("not in table, skipped:", missing)

    fig, axes = plt.subplots(len(SOURCES), len(METRICS), figsize=(2.4 * len(METRICS), 2.4 * len(SOURCES) + 0.8),
                             constrained_layout=True, squeeze=False)
    x = np.arange(len(CATS))
    w = 0.8 / max(len(variants), 1)
    for r, (sk, slabel) in enumerate(SOURCES):
        for c, (m, mlabel, higher) in enumerate(METRICS):
            ax = axes[r, c]
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            for j, (meth, vlabel, color) in enumerate(variants):
                v = df[df.method == meth].set_index(keys)
                vals = []
                for ck, _ in CATS:
                    k = (ck, sk)
                    if k in v.index and k in nat.index:
                        d = float(v.loc[k, f"{m}@{args.turn}"]) - float(nat.loc[k, f"{m}@{args.turn}"])
                        vals.append(d if higher else -d)
                    else:
                        vals.append(np.nan)
                pos = x + (j - (len(variants) - 1) / 2) * w
                ax.bar(pos, vals, w * 0.88, color=color, label=vlabel)
            ax.set_xticks(x, [cl for _, cl in CATS])
            if r == 0:
                ax.set_title(("Δ " if higher else "−Δ ") + mlabel, color=TEXT)
            if c == 0:
                ax.set_ylabel(f"{slabel}\nvariant − native (↑ better)")
    align_zero(axes.ravel())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(variants))
    n = df[df.method == "native"]["n"].sum() if "n" in df else ""
    when = f"cumulative mean over turns 1–{args.turn}" if args.turn > 1 else "turn 1"
    fig.suptitle(args.title or f"Band ablation on {dict(kontext='FLUX.1 Kontext', qwen='Qwen-Image-Edit').get(args.model, args.model)}, gain over native, {when}",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"{args.name}_t{args.turn}.{ext}"), dpi=220, bbox_inches="tight")
    print(f"{args.out}/{args.name}_t{args.turn}.png/.pdf")


if __name__ == "__main__":
    main()
