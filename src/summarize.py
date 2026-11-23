"""Summary tables: model x method x category (and x source), cumulative metrics at turns 1 and 10.

    python summarize.py output_identity output_identity_A1 output_identity_A2 --out tables/identity
"""
import argparse, os, re
import pandas as pd

COLS = ["cum_psnr", "cum_ssim", "cum_lpips", "cum_clip_i", "cum_dino"]
ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--out", required=True, help="prefix for the csv files")
ap.add_argument("--turns", nargs="*", type=int, default=[1, 10])
args = ap.parse_args()

dfs = []
for root in args.roots:
    p = os.path.join(root, "summary.csv")
    if not os.path.exists(p):
        print(f"skip {root}: no summary.csv"); continue
    d = pd.read_csv(p)
    variant = re.sub(r"^output_(nostyle|identity)_?", "", os.path.basename(os.path.normpath(root)))
    d["model"] = d["run"].str.split("_").str[0]
    d["method"] = d["run"].str.split("_", n=1).str[1] + (f"_{variant}" if variant else "")
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df[["category", "source"]] = df["image"].str.split("_", expand=True)[[0, 1]]

def table(keys):
    t = df[df.turn.isin(args.turns)].groupby(keys + ["turn"])[COLS].mean().unstack("turn")
    t.columns = [f"{m[4:]}@{k}" for m, k in t.columns]
    n = df[df.turn == max(args.turns)].groupby(keys)["image"].nunique().rename("n")
    return pd.concat([n, t], axis=1).round(3)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
for name, keys in [("by_category", ["model", "method", "category"]),
                   ("by_category_source", ["model", "method", "category", "source"])]:
    t = table(keys)
    t.to_csv(f"{args.out}_{name}.csv")
    print(f"\n=== {name} ===\n{t.to_string()}")
print(f"\nwritten: {args.out}_by_category.csv, {args.out}_by_category_source.csv")
