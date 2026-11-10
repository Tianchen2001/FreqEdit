"""Generate candidate source images with FLUX.1-dev for the dataset.

Put this file in FreqEdit/src/ (any folder works; it only needs diffusers).
Example:
    python gen_images.py --prompts ../data/gen_prompts.jsonl --out ../data/generated_candidates --seeds 0 1 2 3

Input: JSONL, one {"id", "category", "prompt"} per line.
Output, under --out:
    {category}/{id}_s{seed}.png          1024x1024 candidates
    metadata.jsonl                       id, category, prompt, seed, steps, guidance, model, file
    contact_{category}.jpg               all candidates of a category in one sheet, labelled id_s{seed}
Existing files are skipped, so an interrupted run can be restarted with the same command.
"""
import argparse
import json
import os

import torch
from diffusers import FluxPipeline
from PIL import Image, ImageDraw

MODEL = "black-forest-labs/FLUX.1-dev"


def contact_sheet(paths, out, tile=256, cols=8):
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        x, y = (i % cols) * tile, (i // cols) * (tile + 20)
        sheet.paste(Image.open(p).convert("RGB").resize((tile, tile)), (x, y))
        draw.text((x + 4, y + tile + 3), os.path.splitext(os.path.basename(p))[0], fill="black")
    sheet.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these ids or categories")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.only:
        items = [it for it in items if it["id"] in args.only or it["category"] in args.only]
    os.makedirs(args.out, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    meta_path = os.path.join(args.out, "metadata.jsonl")
    done = set()
    if os.path.exists(meta_path):
        done = {json.loads(l)["file"] for l in open(meta_path) if l.strip()}

    for it in items:
        cat_dir = os.path.join(args.out, it["category"])
        os.makedirs(cat_dir, exist_ok=True)
        for s in args.seeds:
            rel = os.path.join(it["category"], f"{it['id']}_s{s}.png")
            path = os.path.join(args.out, rel)
            if os.path.exists(path):
                continue
            print(f"{it['id']} seed {s}")
            img = pipe(it["prompt"], height=args.size, width=args.size, num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       generator=torch.Generator("cuda").manual_seed(s)).images[0]
            img.save(path)
            if rel not in done:
                with open(meta_path, "a") as f:
                    f.write(json.dumps(dict(id=it["id"], category=it["category"], prompt=it["prompt"], seed=s,
                                            steps=args.steps, guidance=args.guidance, model=MODEL,
                                            size=args.size, file=rel)) + "\n")

    for cat in sorted({it["category"] for it in items}):
        cat_dir = os.path.join(args.out, cat)
        paths = sorted(os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".png"))
        if paths:
            contact_sheet(paths, os.path.join(args.out, f"contact_{cat}.jpg"))
            print(f"contact sheet: contact_{cat}.jpg ({len(paths)} images)")


if __name__ == "__main__":
    main()
