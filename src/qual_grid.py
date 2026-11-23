"""Qualitative figure: for each image, one row per method, one column per turn. All methods in one figure.

Run from FreqEdit/src:
    # the four baselines (default rows) on two images, turns 0 1 3 5 7 10
    python qual_grid.py --root output_nostyle --images smooth_real_05 textured_gen_01 --out figs/qual_nostyle

    # any runs, from any output folder: "label=folder:run"
    python qual_grid.py --images smooth_real_05 --out figs/qual_identity_ablation \
        --rows "Kontext native=output_identity:kontext_native" \
               "FreqEdit=output_identity:kontext_freqedit" \
               "A1 D1+D2=output_identity_A1:kontext_freqedit" \
               "A2 D1=output_identity_A2:kontext_freqedit" \
               "A3 D2=output_identity_A3:kontext_freqedit"

    # zoom: add a second, enlarged crop under each tile (normalized box x0 y0 x1 y1)
    python qual_grid.py --root output_identity --images smooth_real_05 --zoom 0.6 0.1 0.9 0.4 --out figs/qual_zoom

Writes {out}.png and {out}.pdf. A missing turn is left blank.
"""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

DEFAULT_ROWS = ["Kontext native={root}:kontext_native", "Kontext FreqEdit={root}:kontext_freqedit",
                "Qwen native={root}:qwen_native", "Qwen FreqEdit={root}:qwen_freqedit"]
ZOOM_COLOR = "#e5484d"


def parse_rows(specs, root):
    rows = []
    for s in specs:
        label, path = s.rsplit("=", 1)
        folder, run = path.format(root=root).split(":", 1)
        if not os.path.isdir(folder):
            raise SystemExit(f"row '{label}': folder not found: {folder}")
        rows.append((label, folder, run))
    return rows


def load(folder, image, run, turn, size):
    p = os.path.join(folder, image, run, f"turn_{turn:02d}.png")
    if not os.path.exists(p):
        return None
    return Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)


def crop(im_path, box, size):
    im = Image.open(im_path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box
    return im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--root", default="output_nostyle", help="fills {root} in the default rows")
    ap.add_argument("--rows", nargs="*", default=None, help='"label=folder:run" per row, in order')
    ap.add_argument("--turns", nargs="*", type=int, default=[0, 1, 3, 5, 7, 10])
    ap.add_argument("--zoom", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--tile", type=float, default=1.35, help="inches per tile")
    ap.add_argument("--prompts", action="store_true", help="print the prompt of each turn above the columns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = parse_rows(args.rows or DEFAULT_ROWS, args.root)
    sub = 2 if args.zoom else 1                      # tiles per (row, turn): full image (+ zoom)
    per = len(rows) * sub
    n_cols = len(args.turns)
    gap = 0.22                                       # spacer row between images, in tile heights
    ratios = []
    for i in range(len(args.images)):
        if i:
            ratios.append(gap)
        ratios += [1] * per
    fig = plt.figure(figsize=(n_cols * args.tile + 1.4, sum(ratios) * args.tile * 1.06 + (0.6 if args.prompts else 0.2)))
    gs = fig.add_gridspec(len(ratios), n_cols, height_ratios=ratios, wspace=0.03, hspace=0.06)
    axes = {}
    g = 0
    for i in range(len(args.images)):
        if i:
            g += 1
        for k in range(per):
            for c in range(n_cols):
                axes[i * per + k, c] = fig.add_subplot(gs[g, c])
            g += 1
    px = 384

    for i, image in enumerate(args.images):
        prompts = []
        for _, folder, run in rows:
            log = os.path.join(folder, image, run, "log.json")
            if os.path.exists(log):
                prompts = json.load(open(log)).get("prompts", [])
                break
        for r, (label, folder, run) in enumerate(rows):
            base = (i * len(rows) + r) * sub
            for c, t in enumerate(args.turns):
                ax = axes[base, c]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                im = load(folder, image, run, t, px)
                if im is not None:
                    ax.imshow(im)
                    if args.zoom:
                        x0, y0, x1, y1 = args.zoom
                        ax.add_patch(Rectangle((x0 * px, y0 * px), (x1 - x0) * px, (y1 - y0) * px,
                                               fill=False, edgecolor=ZOOM_COLOR, linewidth=0.8))
                if args.zoom:
                    az = axes[base + 1, c]
                    az.set_xticks([]); az.set_yticks([])
                    for sp in az.spines.values():
                        sp.set_edgecolor(ZOOM_COLOR); sp.set_linewidth(0.8)
                    p = os.path.join(folder, image, run, f"turn_{t:02d}.png")
                    if os.path.exists(p):
                        az.imshow(crop(p, args.zoom, px))
                if i == 0 and r == 0:
                    title = "input" if t == 0 else f"turn {t}"
                    if args.prompts and 0 < t <= len(prompts):
                        title += "\n" + "\n".join(textwrap.wrap(prompts[t - 1], max(16, int(args.tile * 19)))[:4])
                    ax.set_title(title, fontsize=6 if args.prompts else 9, pad=3)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=6)
            if r == 0:
                axes[base, 0].annotate(image, xy=(0, 1), xycoords="axes fraction", xytext=(-4, 4 if i else (40 if args.prompts else 16)),
                                       textcoords="offset points", ha="right", va="bottom", fontsize=8,
                                       color="#52514e", annotation_clip=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"{args.out}.png / .pdf")


if __name__ == "__main__":
    main()
