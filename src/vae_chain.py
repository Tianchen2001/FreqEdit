"""VAE-only chain: X -> VAE encode -> VAE decode -> X', repeated, no transformer.

Same preprocessing and the same latent as the editing pipelines (encode with the latent mode, as
retrieve_latents(..., sample_mode="argmax") does; the shift/scale normalisation is skipped because it is
inverted before decoding). Each turn goes through an 8-bit PNG, as in run_chain.py.

Output layout matches run_chain.py, so eval_chain.py / summarize.py work unchanged (method = "vae"):
    {out_root}/{image_name}/{model}_vae/turn_00.png ... turn_10.png, params.json

Run from FreqEdit/src:
    python vae_chain.py --model kontext --images ../data/real/images/*.png ../data/generated/images/*.png --out-root output_vae
"""
import argparse
import json
import os

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

MODEL_IDS = {"kontext": "black-forest-labs/FLUX.1-Kontext-dev", "qwen": "Qwen/Qwen-Image-Edit"}


def load_vae(model):
    if model == "kontext":
        from diffusers import AutoencoderKL as VAE
    else:
        from diffusers import AutoencoderKLQwenImage as VAE
    return VAE.from_pretrained(MODEL_IDS[model], subfolder="vae", torch_dtype=torch.bfloat16).to("cuda").eval()


@torch.no_grad()
def round_trip(vae, proc, img, model, size):
    x = proc.preprocess(img, height=size, width=size).to("cuda", torch.bfloat16)  # [1,3,H,W] in [-1,1]
    if model == "qwen":  # video VAE: [B,C,T,H,W] with T = 1
        z = vae.encode(x.unsqueeze(2)).latent_dist.mode()
        y = vae.decode(z, return_dict=False)[0][:, :, 0]
    else:
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z, return_dict=False)[0]
    return proc.postprocess(y.float(), output_type="pil")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out-root", default="output_vae")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    vae = load_vae(args.model)
    proc = VaeImageProcessor(vae_scale_factor=8)
    for path in args.images:
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_root, name, f"{args.model}_vae")
        if os.path.exists(os.path.join(out, f"turn_{args.turns:02d}.png")):
            continue
        os.makedirs(out, exist_ok=True)
        cur = Image.open(path).convert("RGB").resize((args.size, args.size))  # same resize as run_chain.py
        cur.save(os.path.join(out, "turn_00.png"))
        for k in range(1, args.turns + 1):
            cur = round_trip(vae, proc, cur, args.model, args.size)
            cur.save(os.path.join(out, f"turn_{k:02d}.png"))
            cur = Image.open(os.path.join(out, f"turn_{k:02d}.png")).convert("RGB")  # 8-bit, like the chains
        with open(os.path.join(out, "params.json"), "w") as f:
            json.dump(dict(model=args.model, model_id=MODEL_IDS[args.model], method="vae", latent="mode",
                           turns=args.turns, size=args.size, source_image=os.path.abspath(path)), f, indent=2)
        print("done", out)


if __name__ == "__main__":
    main()
