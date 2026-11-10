"""Build the 1024x1024 dataset from the hand-picked images.

Input (run from data/):
    real/raw/{textured,smooth,human}/{gedit_key}.{ext}       hand-picked GEdit input_image_raw
    gedit_raw/metadata.csv                                    from prep_gedit.py (instructions, task_type)
    generated/selected/{category}/{id}_s{seed}.png            hand-picked FLUX images (optional)
    generated_candidates/{category}/{id}_s{seed}.png          all FLUX candidates + metadata.jsonl
        -> if generated/selected/{category}/ is missing or empty, take --gen-seed of every prompt

Output:
    real/images/{category}_real_{nn}.png         square crop (largest centered square) -> 1024, Lanczos
    generated/images/{category}_gen_{nn}.png     copied (already 1024)
    manifest.csv                                 one row per image, with origin and crop box
    overview/crops_{category}.jpg                originals with the crop box drawn
    overview/final_{category}.jpg                final 1024 images, labelled with the new id

crops.json (optional) moves the square crop for a real image:
    {"<gedit_key>": 0.3}     position along the long side, 0 = left/top, 0.5 = center (default), 1 = right/bottom

Usage:
    python build_dataset.py                     # dry run: prints the plan, writes nothing
    python build_dataset.py --write
    python build_dataset.py --write --crops crops.json --overwrite
"""
import argparse
import csv
import glob
import json
import os
import shutil

from PIL import Image, ImageDraw, ImageOps

CATS = ["textured", "smooth", "human"]
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SIZE = 1024


def list_images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(EXTS))


def square_box(w, h, pos=0.5):
    s = min(w, h)
    if w >= h:
        x0 = int(round((w - s) * pos))
        return (x0, 0, x0 + s, s)
    y0 = int(round((h - s) * pos))
    return (0, y0, s, y0 + s)


def grid(tiles, labels, cols=6, tile=256):
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(tiles, labels)):
        im = im.copy()
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 20)
        sheet.paste(im, (x + (tile - im.width) // 2, y))
        draw.text((x + 4, y + tile + 3), lab[:40], fill="black")
    return sheet


def load_gedit_meta(path):
    if not os.path.exists(path):
        print(f"warning: {path} not found, instructions left empty")
        return {}
    with open(path) as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def load_gen_meta(path):
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    meta[os.path.basename(r["file"])] = r
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="the data/ folder")
    ap.add_argument("--crops", default=None)
    ap.add_argument("--gen-seed", type=int, default=0, help="seed to use when generated/selected/{cat} is empty")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    R = lambda *p: os.path.join(args.root, *p)

    crops = json.load(open(args.crops)) if args.crops else {}
    gedit = load_gedit_meta(R("gedit_raw", "metadata.csv"))
    gen_meta = load_gen_meta(R("generated_candidates", "metadata.jsonl"))

    out_real, out_gen, out_ov = R("real", "images"), R("generated", "images"), R("overview")
    if args.write:
        for d in (out_real, out_gen):
            if list_images(d) and not args.overwrite:
                raise FileExistsError(f"{d} is not empty; use --overwrite")
            if args.overwrite and os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        os.makedirs(out_ov, exist_ok=True)

    manifest, used_crops = [], set()
    for cat in CATS:
        # ---------------- real
        raw = list_images(R("real", "raw", cat))
        orig_tiles, final_tiles, labels = [], [], []
        for n, path in enumerate(raw, 1):
            key = os.path.splitext(os.path.basename(path))[0]
            new_id = f"{cat}_real_{n:02d}"
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            w, h = im.size
            pos = float(crops.get(key, 0.5))
            used_crops.add(key) if key in crops else None
            box = square_box(w, h, pos)
            note = "upscaled" if min(w, h) < SIZE else ""
            out = im.crop(box).resize((SIZE, SIZE), Image.LANCZOS)
            g = gedit.get(key, {})
            manifest.append(dict(
                id=new_id, category=cat, source="real", file=f"real/images/{new_id}.png",
                origin_file=os.path.relpath(path, args.root), origin_key=key,
                orig_w=w, orig_h=h, crop_box=" ".join(map(str, box)), crop_pos=pos, note=note,
                gedit_task_type=g.get("task_type", ""), gedit_instruction_en=g.get("instruction_en", ""),
                gen_prompt="", gen_seed=""))
            print(f"{new_id:20s} <- {key}  {w}x{h}  crop {box}  {note}")
            if args.write:
                out.save(os.path.join(out_real, f"{new_id}.png"))
                vis = im.copy()
                ImageDraw.Draw(vis).rectangle(box, outline="red", width=max(4, min(w, h) // 100))
                orig_tiles.append(vis)
                final_tiles.append(out)
                labels.append(f"{new_id} {key[:12]}")

        # ---------------- generated
        sel = list_images(R("generated", "selected", cat))
        if not sel:
            sel = list_images(R("generated_candidates", cat))
            sel = [p for p in sel if p.endswith(f"_s{args.gen_seed}.png")]
            if sel:
                print(f"[{cat}] generated/selected empty -> using seed {args.gen_seed} of every prompt")
        g_tiles, g_labels = [], []
        for n, path in enumerate(sel, 1):
            fname = os.path.basename(path)
            new_id = f"{cat}_gen_{n:02d}"
            im = Image.open(path).convert("RGB")
            if im.size != (SIZE, SIZE):
                im = im.crop(square_box(*im.size)).resize((SIZE, SIZE), Image.LANCZOS)
            m = gen_meta.get(fname, {})
            manifest.append(dict(
                id=new_id, category=cat, source="generated", file=f"generated/images/{new_id}.png",
                origin_file=os.path.relpath(path, args.root), origin_key=m.get("id", fname.rsplit("_s", 1)[0]),
                orig_w=im.width, orig_h=im.height, crop_box="", crop_pos="", note="",
                gedit_task_type="", gedit_instruction_en="",
                gen_prompt=m.get("prompt", ""), gen_seed=m.get("seed", fname.rsplit("_s", 1)[-1][:-4])))
            print(f"{new_id:20s} <- {fname}")
            if args.write:
                im.save(os.path.join(out_gen, f"{new_id}.png"))
                g_tiles.append(im)
                g_labels.append(f"{new_id} {fname[:20]}")

        if args.write:
            if orig_tiles:
                grid(orig_tiles, labels).save(os.path.join(out_ov, f"crops_{cat}.jpg"), quality=90)
            if final_tiles or g_tiles:
                grid(final_tiles + g_tiles, labels + g_labels).save(os.path.join(out_ov, f"final_{cat}.jpg"), quality=90)

    unused = set(crops) - used_crops
    if unused:
        print(f"warning: crops.json keys not found in real/raw: {sorted(unused)}")
    counts = {(c, s): sum(r["category"] == c and r["source"] == s for r in manifest) for c in CATS for s in ("real", "generated")}
    print("\ncounts:", ", ".join(f"{c}/{s}={n}" for (c, s), n in counts.items()))
    if args.write and manifest:
        with open(R("manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0]))
            w.writeheader()
            w.writerows(manifest)
        print(f"wrote {R('manifest.csv')} and {out_ov}/")
    elif not args.write:
        print("dry run; add --write")


if __name__ == "__main__":
    main()
