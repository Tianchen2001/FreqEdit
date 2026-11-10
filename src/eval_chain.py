"""Per-turn metrics for chains produced by run_chain.py.

Put this file in FreqEdit/src/ next to run_chain.py.

Default: the FreqEdit similarity metrics, whole image, edited X[k+1] vs the source X[1]
(X[1] = turn_00.png, X[k+1] = turn_{k:02d}.png):
    psnr, ssim, dino      (FreqEdit App. D, compared with the original image X[1])
    lpips, clip_i         (FreqEdit Sec. 5.1, "edited vs source"; we also use X[1])
plus their cumulative mean over turns 1..k (cum_*), the form reported in FreqEdit Table 2.
The GPT-4o based metrics of FreqEdit (instruction following, consistency, quality) are not here.

--extra adds our own metrics (not in FreqEdit):
    psnr_prev, lpips_prev                    vs the previous turn X[k]
    ll_mse, hf_mse, hf_energy, hf_ratio      2-level db4 on luminance; hf_ratio < 1 detail loss, > 1 noise gain
    {region}_*                               the same, inside each box of --regions (whole-image columns stay)

Usage:
    python eval_chain.py "output/*/*" --summary output/summary.csv
    python eval_chain.py "output/*/*" --extra --regions regions.json --summary output/summary_extra.csv

Writes metrics.json / metrics.csv in every run folder, and with --summary:
    summary.csv         every image, run and turn in one table
    summary_table.csv   per run, averaged over images: cumulative mean at turns 1, 4, 7, 10 (FreqEdit Table 2 layout)

regions.json maps the image folder name to boxes in normalized [x0, y0, x1, y1]:
    {"img_1": {"grass": [0.02, 0.78, 0.30, 0.98]}, "img_4": {"background": [0.60, 0.02, 0.98, 0.22]}}
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np
import pywt
from PIL import Image
from skimage.metrics import structural_similarity

WAVELET, LEVELS = "db4", 2


# ---------------------------------------------------------------- basic ops
def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def bands(gray):
    """2-level db4 decomposition -> (LL, list of detail arrays)."""
    coeffs = pywt.wavedec2(gray, WAVELET, level=LEVELS, mode="symmetric")
    return coeffs[0], [d for level in coeffs[1:] for d in level]


def band_metrics(x, ref):
    ll_x, hf_x = bands(luma(x))
    ll_r, hf_r = bands(luma(ref))
    ll_mse = float(np.mean((ll_x - ll_r) ** 2))
    hf_mse = float(np.mean(np.concatenate([(a - b).ravel() ** 2 for a, b in zip(hf_x, hf_r)])))
    n = sum(d.size for d in hf_x)
    e_x = sum(float(np.sum(d ** 2)) for d in hf_x)
    e_r = sum(float(np.sum(d ** 2)) for d in hf_r)
    # hf_energy is reported too: on very flat regions e_r is tiny and the ratio alone exaggerates
    return dict(ll_mse=ll_mse, hf_mse=hf_mse, hf_energy=e_x / n,
                hf_ratio=e_x / e_r if e_r > 0 else float("nan"))


def crop(x, box):
    h, w = x.shape[:2]
    x0, y0, x1, y1 = box
    return x[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))]


# ---------------------------------------------------------------- deep metrics
class Deep:
    """LPIPS, CLIP-I and DINO similarity; each can be switched off."""

    def __init__(self, device, use_lpips=True, use_clip=True, use_dino=True):
        self.device = device
        self.lpips = self.clip = self.dino = None
        if use_lpips or use_clip or use_dino:
            import torch
            self.torch = torch
        if use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        if use_clip:
            from transformers import CLIPModel, CLIPProcessor
            name = "openai/clip-vit-large-patch14"
            self.clip = CLIPModel.from_pretrained(name).to(device).eval()
            self.clip_proc = CLIPProcessor.from_pretrained(name)
        if use_dino:
            from transformers import AutoImageProcessor, AutoModel
            name = "facebook/dinov2-base"
            self.dino = AutoModel.from_pretrained(name).to(device).eval()
            self.dino_proc = AutoImageProcessor.from_pretrained(name)

    def _t(self, x):
        t = self.torch.from_numpy(x).float().permute(2, 0, 1)[None].to(self.device)
        return t * 2 - 1  # LPIPS expects [-1, 1]

    def lpips_dist(self, a, b):
        if self.lpips is None or min(a.shape[:2]) < 32:
            return float("nan")
        with self.torch.no_grad():
            return float(self.lpips(self._t(a), self._t(b)).item())

    def _embed(self, model, proc, a, b, kind):
        ims = [Image.fromarray((v * 255).round().astype(np.uint8)) for v in (a, b)]
        inp = proc(images=ims, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            if kind == "clip":
                f = model.get_image_features(**inp)
            else:
                f = model(**inp).last_hidden_state[:, 0]  # CLS token
        f = self.torch.nn.functional.normalize(f, dim=-1)
        return float((f[0] * f[1]).sum().item())

    def clip_sim(self, a, b):
        return float("nan") if self.clip is None else self._embed(self.clip, self.clip_proc, a, b, "clip")

    def dino_sim(self, a, b):
        return float("nan") if self.dino is None else self._embed(self.dino, self.dino_proc, a, b, "dino")


# ---------------------------------------------------------------- per run
PAPER = ["psnr", "ssim", "lpips", "clip_i", "dino"]


def eval_run(run_dir, deep, regions, extra=False):
    turns = sorted(glob.glob(os.path.join(run_dir, "turn_*.png")))
    if len(turns) < 2 or not turns[0].endswith("turn_00.png"):
        raise FileNotFoundError(f"{run_dir}: need turn_00.png and at least one edited turn")
    src = load(turns[0])
    prev = src
    rows = []
    for path in turns[1:]:
        k = int(os.path.basename(path)[5:7])
        x = load(path)
        if x.shape != src.shape:
            raise ValueError(f"{path}: shape {x.shape} != source {src.shape}")
        row = dict(turn=k,
                   psnr=psnr(x, src),
                   ssim=float(structural_similarity(x, src, channel_axis=2, data_range=1.0)),
                   lpips=deep.lpips_dist(x, src),
                   clip_i=deep.clip_sim(x, src),
                   dino=deep.dino_sim(x, src))
        if extra:
            row.update(psnr_prev=psnr(x, prev), lpips_prev=deep.lpips_dist(x, prev))
            row.update(band_metrics(x, src))
            for name, box in regions.items():
                xc, sc = crop(x, box), crop(src, box)
                row[f"{name}_psnr"] = psnr(xc, sc)
                row[f"{name}_lpips"] = deep.lpips_dist(xc, sc)
                for key, val in band_metrics(xc, sc).items():
                    row[f"{name}_{key}"] = val
        rows.append(row)
        prev = x
        print(f"  turn {k:2d}  psnr {row['psnr']:.2f}  ssim {row['ssim']:.3f}  lpips {row['lpips']:.3f}  "
              f"clip_i {row['clip_i']:.3f}  dino {row['dino']:.3f}")
    # cumulative mean over turns 1..k, as in FreqEdit Table 2
    for i, row in enumerate(rows):
        for m in PAPER:
            vals = [r[m] for r in rows[: i + 1] if not math.isnan(r[m]) and not math.isinf(r[m])]
            row[f"cum_{m}"] = float(np.mean(vals)) if vals else float("nan")
    return rows


def paper_table(summary, report_turns=(1, 4, 7, 10)):
    """Per run: cumulative mean at the report turns, averaged over images."""
    out = []
    for run in sorted({r["run"] for r in summary}):
        row = dict(run=run, n_images=len({r["image"] for r in summary if r["run"] == run}))
        for m in PAPER:
            for t in report_turns:
                vals = [r[f"cum_{m}"] for r in summary if r["run"] == run and r["turn"] == t
                        and not math.isnan(r[f"cum_{m}"])]
                row[f"{m}@{t}"] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def write_csv(path, rows, extra=None):
    keys = list((extra or {}).keys()) + list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({**(extra or {}), **r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, e.g. output/img_4/kontext_native (globs ok)")
    ap.add_argument("--regions", default=None, help="json: image folder name -> {region: [x0,y0,x1,y1]}")
    ap.add_argument("--summary", default=None, help="summary csv; summary_table.csv is written next to it")
    ap.add_argument("--extra", action="store_true", help="also compute our band / region metrics (not in FreqEdit)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--no-dino", action="store_true")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)})
    all_regions = json.load(open(args.regions)) if args.regions else {}
    deep = Deep(args.device, not args.no_lpips, not args.no_clip, not args.no_dino)

    summary = []
    for run_dir in run_dirs:
        image = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
        run = os.path.basename(os.path.normpath(run_dir))
        regions = all_regions.get(image, {})
        print(f"{image}/{run}" + (f"  regions: {list(regions)}" if args.extra and regions else ""))
        rows = eval_run(run_dir, deep, regions, args.extra)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(dict(image=image, run=run, extra=args.extra, regions=regions if args.extra else {},
                           turns=rows), f, indent=2)
        write_csv(os.path.join(run_dir, "metrics.csv"), rows)
        summary += [{"image": image, "run": run, **r} for r in rows]

    if args.summary and summary:
        write_csv(args.summary, summary)
        table_path = os.path.join(os.path.dirname(args.summary) or ".", "summary_table.csv")
        write_csv(table_path, paper_table(summary))
        print(f"summary: {args.summary}\ntable:   {table_path}")


if __name__ == "__main__":
    main()