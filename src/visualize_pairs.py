"""Native vs FreqEdit sheets for every image and every model, for picking qualitative examples.

Put in FreqEdit/src/. Run from there:
    python visualize_pairs.py output_nostyle output_identity
    python visualize_pairs.py output_nostyle --models kontext --turns-shown 0 1 3 5 7 10 --tile 384

For each out_root, model and image, writes
    {out_root}/_pairs/{model}/{image}.jpg
        row 1: {model}_native     turn 0 .. 10
        row 2: {model}_freqedit   turn 0 .. 10
        under each tile: PSNR / LPIPS vs turn 0 (from metrics.json, if eval_chain.py was run)
        header: image, category / source, cumulative PSNR and LPIPS @10 of both runs and the gain
        footer: the editing prompts
    {out_root}/_pairs/{model}/index.html   all sheets on one page, sorted by FreqEdit gain (smallest first)
"""
import argparse
import html
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return {}
    return {r["turn"]: r for r in json.load(open(p))["turns"]}


def load_prompts(run_dir):
    p = os.path.join(run_dir, "log.json")
    return json.load(open(p)).get("prompts", []) if os.path.exists(p) else []


def sheet(image, runs, turns, tile, prompts):
    """runs: list of (label, run_dir, metrics)."""
    f_s, f_m, f_b = font(14), font(16), font(20)
    label_w, head, sub = 170, 76, 22
    W = label_w + len(turns) * tile
    lines = [w for k, p in enumerate(prompts, 1) for w in textwrap.wrap(f"{k}. {p}", width=max(60, W // 8))]
    H = head + len(runs) * (tile + sub) + 12 + len(lines) * 18 + 10
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    cat, src = (image.split("_") + ["", ""])[:2]
    head_txt = f"{image}   ({cat} / {src})"
    stats = []
    for label, _, m in runs:
        last = m.get(max(m)) if m else None
        if last:
            stats.append(f"{label}: cumPSNR {last['cum_psnr']:.2f}  cumLPIPS {last['cum_lpips']:.3f}")
    d.text((8, 4), head_txt, fill="black", font=f_b)
    d.text((8, 30), "    ".join(stats), fill=(60, 60, 60), font=f_s)
    for j, k in enumerate(turns):
        d.text((label_w + j * tile + 4, head - 18), f"turn {k}", fill="black", font=f_s)

    for row, (label, run_dir, m) in enumerate(runs):
        y = head + row * (tile + sub)
        d.text((6, y + tile // 2 - 9), label, fill="black", font=f_m)
        for j, k in enumerate(turns):
            x = label_w + j * tile
            p = os.path.join(run_dir, f"turn_{k:02d}.png")
            if os.path.exists(p):
                im.paste(Image.open(p).convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
            else:
                d.rectangle([x, y, x + tile - 1, y + tile - 1], outline="gray")
            if k in m:
                d.text((x + 4, y + tile + 3), f"{m[k]['psnr']:.1f} dB  {m[k]['lpips']:.3f}", fill=(60, 60, 60), font=f_s)

    y = head + len(runs) * (tile + sub) + 12
    for l in lines:
        d.text((8, y), l, fill="black", font=f_s)
        y += 18
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_roots", nargs="+")
    ap.add_argument("--models", nargs="*", default=["kontext", "qwen"])
    ap.add_argument("--turns-shown", nargs="*", type=int, default=list(range(11)))
    ap.add_argument("--tile", type=int, default=256)
    args = ap.parse_args()

    for root in args.out_roots:
        images = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
        for model in args.models:
            dst = os.path.join(root, "_pairs", model)
            entries = []
            for image in images:
                runs = []
                for meth in ("native", "freqedit"):
                    rd = os.path.join(root, image, f"{model}_{meth}")
                    if os.path.exists(os.path.join(rd, "turn_00.png")):
                        runs.append((f"{model}_{meth}", rd, load_metrics(rd)))
                if len(runs) < 2:
                    continue
                os.makedirs(dst, exist_ok=True)
                prompts = load_prompts(runs[1][1]) or load_prompts(runs[0][1])
                sheet(image, runs, args.turns_shown, args.tile, prompts).save(os.path.join(dst, f"{image}.jpg"), quality=88)
                # FreqEdit gain in cumulative LPIPS at the last turn (positive = FreqEdit better)
                gain = None
                mn, mf = runs[0][2], runs[1][2]
                if mn and mf and max(mn) == max(mf):
                    gain = mn[max(mn)]["cum_lpips"] - mf[max(mf)]["cum_lpips"]
                entries.append((image, gain))
                print(f"{root} {model} {image}  gain_lpips={gain if gain is None else round(gain, 3)}")
            if not entries:
                continue
            entries.sort(key=lambda e: (e[1] is None, e[1] if e[1] is not None else 0))
            with open(os.path.join(dst, "index.html"), "w") as f:
                f.write(f"<html><head><meta charset='utf-8'><title>{html.escape(root)} {model}</title>"
                        "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%;border:1px solid #ccc}"
                        "h3{margin:28px 0 6px}</style></head><body>")
                f.write(f"<h2>{html.escape(root)} / {model}: native vs FreqEdit, sorted by LPIPS gain (smallest first)</h2>")
                for image, gain in entries:
                    g = "n/a" if gain is None else f"{gain:+.3f}"
                    f.write(f"<h3>{html.escape(image)} &nbsp; gain {g}</h3><img src='{html.escape(image)}.jpg' loading='lazy'>")
                f.write("</body></html>")
            print(f"-> {dst}/index.html")


if __name__ == "__main__":
    main()
