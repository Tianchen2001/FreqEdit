"""Per-turn tone/colour statistics of a chain, relative to turn 0 (a 'grey veil' shows as lifted blacks,
lower contrast and lower chroma).

For each run folder and turn k (computed on downsampled 256x256 images, CIELAB, D65):
    L_mean, L_std (contrast), C_mean (chroma), L_p01 / L_p99 (black / white point)
    gain_c, off_c  per RGB channel: least-squares fit  X[k]_c ~= gain * X[0]_c + off   (pixels in [0,1])
                   veil = gain < 1 with off > 0. Only meaningful where content is unchanged
                   (VAE and identity chains); on editing chains read L/C stats instead.
    noise part, on the full-resolution luminance Y (in [0,1]) of the residual R = Y[k] - Y[0]:
        res_low_rms  = RMS of gaussian(R, sigma)         tone / colour drift (the veil)
        res_high_rms = RMS of R - gaussian(R, sigma)     fine-scale change (noise, or lost detail)
        hf_ratio     = HF energy of Y[k] / HF energy of Y[0], HF = Y - gaussian(Y, sigma)
                       > 1 adds fine-scale energy (noise), < 1 removes it (smoothing)
Writes one row per (run, turn); the d_* columns are the change from turn 0.

Run from FreqEdit/src:
    python color_drift.py "output_vae/*/*" "output_identity/*/*" --out tables/color_drift.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter


def srgb_to_lab(rgb):
    a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = a @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def load(p, size=256):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float64) / 255.0


def luma(p):
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
    return x @ np.array([0.2126, 0.7152, 0.0722])


def noise_stats(y, y0, sigma):
    r = y - y0
    r_low = gaussian_filter(r, sigma)
    hf = lambda z: np.mean((z - gaussian_filter(z, sigma)) ** 2)
    return dict(res_low_rms=np.sqrt(np.mean(r_low ** 2)), res_high_rms=np.sqrt(np.mean((r - r_low) ** 2)),
                hf_ratio=hf(y) / max(hf(y0), 1e-12))


def stats(rgb, ref):
    lab = srgb_to_lab(rgb)
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    out = dict(L_mean=L.mean(), L_std=L.std(), C_mean=C.mean(), L_p01=np.percentile(L, 1), L_p99=np.percentile(L, 99))
    for i, c in enumerate("rgb"):
        A = np.stack([ref[..., i].ravel(), np.ones(ref[..., i].size)], 1)
        g, o = np.linalg.lstsq(A, rgb[..., i].ravel(), rcond=None)[0]
        out[f"gain_{c}"], out[f"off_{c}"] = g, o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help='run folders, e.g. "output_vae/*/*"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=2.0, help="gaussian sigma in pixels at full resolution")
    args = ap.parse_args()

    rows = []
    for pat in args.patterns:
        for d in sorted(glob.glob(pat)):
            turns = sorted(glob.glob(os.path.join(d, "turn_*.png")))
            if len(turns) < 2:
                continue
            image, run = d.rstrip("/").split("/")[-2:]
            ref = load(turns[0])
            y0 = luma(turns[0])
            base = stats(ref, ref)
            for t in turns:
                k = int(os.path.basename(t)[5:7])
                s = stats(load(t), ref)
                s.update(noise_stats(luma(t), y0, args.sigma))
                rows.append(dict(folder=d.rstrip("/").split("/")[-3], image=image, run=run, turn=k,
                                 category=image.split("_")[0], source=image.split("_")[1],
                                 **s, **{f"d_{m}": s[m] - base[m] for m in ("L_mean", "L_std", "C_mean", "L_p01", "L_p99")}))
            print(d)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
