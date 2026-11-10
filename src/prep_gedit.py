"""Download GEdit-Bench and extract the raw (full-resolution) source images.

Example (no GPU needed):
    python prep_gedit.py --out ../data/gedit_raw --min-side 1024

Output, under --out:
    images/{task_type}/{key}.{ext}     original bytes of input_image_raw, one file per unique image
    metadata.csv                       key, task_type, width, height, short_side, keep, file, instruction_en, instruction_cn
    contact_{task_type}.jpg            thumbnails of the kept images (short side >= --min-side), labelled by key
"""
import argparse
import csv
import hashlib
import io
import os
from collections import defaultdict

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image, ImageDraw


def contact_sheet(items, out, tile=200, cols=10):
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(items):
        x, y = (i % cols) * tile, (i // cols) * (tile + 18)
        im = Image.open(path).convert("RGB")
        im.thumbnail((tile, tile))
        sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.text((x + 4, y + tile + 2), label, fill="black")
    sheet.save(out, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-side", type=int, default=1024)
    args = ap.parse_args()

    ds = load_dataset("stepfun-ai/GEdit-Bench", split="train")
    ds = ds.cast_column("input_image_raw", HFImage(decode=False))
    ds = ds.remove_columns([c for c in ds.column_names if c == "input_image"])

    by_hash = {}                      # image md5 -> record
    instr = defaultdict(dict)         # image md5 -> {lang: instruction}
    for row in ds:
        raw = row["input_image_raw"]
        data = raw["bytes"] if raw.get("bytes") else open(raw["path"], "rb").read()
        h = hashlib.md5(data).hexdigest()
        instr[h][row["instruction_language"]] = row["instruction"]
        if h in by_hash:
            continue
        im = Image.open(io.BytesIO(data))
        ext = (im.format or "PNG").lower().replace("jpeg", "jpg")
        rel = os.path.join("images", row["task_type"], f"{row['key']}.{ext}")
        path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        w, h_px = im.size
        by_hash[h] = dict(key=row["key"], task_type=row["task_type"], width=w, height=h_px,
                          short_side=min(w, h_px), keep=min(w, h_px) >= args.min_side, file=rel)

    rows = []
    for h, rec in by_hash.items():
        rec["instruction_en"] = instr[h].get("en", "")
        rec["instruction_cn"] = instr[h].get("cn", "")
        rows.append(rec)
    rows.sort(key=lambda r: (r["task_type"], r["key"]))
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(ds)} rows -> {len(rows)} unique images, {sum(r['keep'] for r in rows)} with short side >= {args.min_side}")
    for t in sorted({r["task_type"] for r in rows}):
        kept = [r for r in rows if r["task_type"] == t and r["keep"]]
        print(f"  {t:20s} {sum(r['task_type'] == t for r in rows):4d} images, {len(kept):4d} kept")
        if kept:
            contact_sheet([(os.path.join(args.out, r["file"]), r["key"][:8]) for r in kept],
                          os.path.join(args.out, f"contact_{t}.jpg"))


if __name__ == "__main__":
    main()
