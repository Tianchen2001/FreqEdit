"""Bar charts of the FreqEdit baseline results, from the summarize.py tables.

    python plot_results.py --tables tables --out figs

Reads {tables}/{nostyle,identity}_by_category[_source].csv and writes, for each set (nostyle, identity):
    {out}/{set}_gain_t{turn}.png/.pdf      FreqEdit - native per model and category (LPIPS sign flipped: up = better)
    {out}/{set}_values_t{turn}.png/.pdf    native and FreqEdit values per model and category
    {out}/{set}_gain_source_t{turn}.png/.pdf   gain split into real and generated images
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
MODELS = [("kontext", "FLUX.1 Kontext", "#2a78d6"), ("qwen", "Qwen-Image-Edit", "#eb6834")]
CATS = [("textured", "Textured"), ("smooth", "Smooth"), ("human", "Human")]
SOURCES = [("real", "real"), ("gen", "gen")]
SET_NAMES = {"nostyle": "editing chains (48 images)", "identity": "identity chains (12 images)"}

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "normal", "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def gains(df, keys, turn):
    """FreqEdit - native for each group; LPIPS flipped so that positive = FreqEdit better."""
    nat = df[df.method == "native"].set_index(keys)
    fe = df[df.method == "freqedit"].set_index(keys)
    out = pd.DataFrame(index=fe.index)
    for m, _, higher in METRICS:
        d = fe[f"{m}@{turn}"] - nat[f"{m}@{turn}"]
        out[m] = d if higher else -d
    return out


def style_axis(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def plot_gain(df, keys, groups, turn, title, out, name):
    g = gains(df, keys, turn)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.3 * len(METRICS), 2.9), constrained_layout=True)
    x = np.arange(len(groups))
    w = 0.36
    for ax, (m, label, higher) in zip(axes, METRICS):
        style_axis(ax)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for j, (mk, mlabel, color) in enumerate(MODELS):
            vals = [g.loc[(mk, *grp[0]), m] if (mk, *grp[0]) in g.index else np.nan for grp in groups]
            pos = x + (j - 0.5) * (w + 0.04)
            ax.bar(pos, vals, w, color=color, edgecolor=SURFACE, linewidth=1, label=mlabel)
            for p, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.annotate(f"{v:+.2f}" if m == "psnr" else f"{v:+.3f}".replace("0.", "."),
                                (p, v), xytext=(0, 2 if v >= 0 else -2), textcoords="offset points",
                                ha="center", va="bottom" if v >= 0 else "top", fontsize=6, color=MUTED)
        ax.set_xticks(x, [grp[1] for grp in groups], rotation=0 if len(groups) <= 3 else 35,
                      ha="center" if len(groups) <= 3 else "right")
        ax.set_title(("Δ " if higher else "−Δ ") + label, color=TEXT)
        lo, hi = ax.get_ylim()
        pad = 0.18 * (hi - lo)
        ax.set_ylim(min(lo - pad, -pad), hi + pad)
    axes[0].set_ylabel("FreqEdit − native  (↑ better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    save(fig, out, name)


def align_zero(axes, pad=0.08):
    """Give every axis the same position of y = 0 (same share of its height below zero)."""
    spans = []
    for ax in axes:
        lo, hi = ax.get_ylim()
        neg, pos = max(-lo, 0.0), max(hi, 0.0)
        spans.append((neg, pos))
    frac = min(max([n / (n + p) for n, p in spans if n + p > 0] + [0.0]), 0.9)   # share of height below zero
    for ax, (neg, pos) in zip(axes, spans):
        total = max(neg / frac if frac > 0 else 0, pos / (1 - frac)) * (1 + pad)
        ax.set_ylim(-frac * total, (1 - frac) * total)


def plot_gain_source(df, turn, title, out, name):
    g = gains(df, ["model", "category", "source"], turn)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.3 * len(METRICS), 2.9), constrained_layout=True)
    x = np.arange(len(CATS))
    w = 0.19
    bars = [(mk, mlabel, color, sk) for mk, mlabel, color in MODELS for sk in ("real", "gen")]
    for ax, (m, label, higher) in zip(axes, METRICS):
        style_axis(ax)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for j, (mk, mlabel, color, sk) in enumerate(bars):
            vals = [g.loc[(mk, ck, sk), m] if (mk, ck, sk) in g.index else np.nan for ck, _ in CATS]
            pos = x + (j - 1.5) * (w + 0.02)
            real = sk == "real"
            ax.bar(pos, vals, w, color=SURFACE if real else color, edgecolor=color, linewidth=1.2,
                   hatch="////" if real else None, label=f"{mlabel}, {'real' if real else 'generated'} images")
        ax.set_xticks(x, [c[1] for c in CATS])
        ax.set_title(("Δ " if higher else "−Δ ") + label, color=TEXT)
    align_zero(axes)
    axes[0].set_ylabel("FreqEdit − native  (↑ better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    save(fig, out, name)


def plot_values(df, turn, title, out, name):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.3 * len(METRICS), 2.9), constrained_layout=True)
    x = np.arange(len(CATS))
    w = 0.19
    bars = [(mk, mlabel, color, meth) for mk, mlabel, color in MODELS for meth in ("native", "freqedit")]
    for ax, (m, label, higher) in zip(axes, METRICS):
        style_axis(ax)
        for j, (mk, mlabel, color, meth) in enumerate(bars):
            vals = []
            for ck, _ in CATS:
                r = df[(df.model == mk) & (df.method == meth) & (df.category == ck)]
                vals.append(float(r[f"{m}@{turn}"].iloc[0]) if len(r) else np.nan)
            pos = x + (j - 1.5) * (w + 0.02)
            native = meth == "native"
            ax.bar(pos, vals, w, color=SURFACE if native else color, edgecolor=color,
                   linewidth=1.2, hatch="////" if native else None,
                   label=f"{mlabel}, {'native' if native else 'FreqEdit'}")
        ax.set_xticks(x, [c[1] for c in CATS])
        ax.set_title(f"{label} {'↑' if higher else '↓'}", color=TEXT)
        ax.set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--turns", nargs="*", type=int, default=[10])
    ap.add_argument("--all", action="store_true", help="also the per-category gain and value charts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams["hatch.linewidth"] = 0.6

    for s in ("nostyle", "identity"):
        cat = pd.read_csv(os.path.join(args.tables, f"{s}_by_category.csv"))
        src = pd.read_csv(os.path.join(args.tables, f"{s}_by_category_source.csv"))
        print(s)
        for t in args.turns:
            when = f"cumulative mean over turns 1–{t}" if t > 1 else "turn 1"
            if args.all:
                plot_gain(cat, ["model", "category"], [((ck,), cl) for ck, cl in CATS], t,
                          f"FreqEdit gain over native, {SET_NAMES[s]}, {when}", args.out, f"{s}_gain_t{t}")
                plot_values(cat, t, f"Native vs FreqEdit, {SET_NAMES[s]}, {when}", args.out, f"{s}_values_t{t}")
            plot_gain_source(src, t, f"FreqEdit gain over native by image source, {SET_NAMES[s]}, {when}",
                             args.out, f"{s}_gain_source_t{t}")


if __name__ == "__main__":
    main()
