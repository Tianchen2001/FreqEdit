"""Unified multi-turn editing runner for FreqEdit (Kontext / Qwen, native / FreqEdit).

Replaces run_FreqEditKontext.py, run_FreqEditQwen.py, run_nativeKontext.py, run_nativeQwen.py.
Put this file in FreqEdit/src/ and run from there.

Output layout:
    {out_root}/{image_name}/{model}_{method}[_{tag}]/
        turn_00.png        input image (after resize to 1024x1024)
        turn_01.png ... turn_10.png
        params.json        every parameter actually passed to the pipeline, plus versions and command
        log.json           prompts, seeds and time per turn

Examples:
    python run_chain.py --model kontext --method native   --image ../example_data/img_1.png --prompt-idx 1
    python run_chain.py --model kontext --method freqedit --image ../example_data/img_4.png --prompt-file prompts_pears.txt
    python run_chain.py --model kontext --method freqedit --image ../example_data/img_4.png --prompt-file prompts_pears.txt \
        --qg-lambda 0 --tag noQG
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import torch
from diffusers.utils import load_image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# FreqEdit defaults used in the released run scripts (the pipeline signatures have different defaults)
DEFAULTS = {
    "kontext": dict(inject_time_scale=0.3, base_alpha=1.6, gamma=2.0, qg_time_scale=0.7, qg_lambda=0.3),
    "qwen": dict(inject_time_scale=0.3, base_alpha=2.0, gamma=1.6),
}
MODEL_IDS = {"kontext": "black-forest-labs/FLUX.1-Kontext-dev", "qwen": "Qwen/Qwen-Image-Edit"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["kontext", "qwen"], required=True)
    ap.add_argument("--method", choices=["native", "freqedit"], required=True)

    # input
    ap.add_argument("--image", required=True, help="source image X^[1]; resized to 1024x1024 as in the original scripts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt-file", help="text file, one instruction per line")
    g.add_argument("--prompt-idx", type=int, help="1-based index into example_data/param_prompts.py")

    # output
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--name", default=None, help="image folder name (default: image file stem)")
    ap.add_argument("--tag", default="", help="suffix for the run folder, e.g. noQG, mid30")
    ap.add_argument("--overwrite", action="store_true")

    # turns and seeds
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--start-turn", type=int, default=0, help="resume: continue from turn_{start}.png in the run folder")
    ap.add_argument("--seed", type=int, default=0, help="turn k uses seed + k")

    # sampling
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5, help="Kontext distilled guidance (ignored for Qwen)")
    ap.add_argument("--size", type=int, default=1024)

    # FreqEdit (None -> per-model default above)
    ap.add_argument("--inject-time-scale", type=float, default=None)
    ap.add_argument("--base-alpha", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--no-adaptive-weight", action="store_true")
    ap.add_argument("--backward-steps", type=int, default=4)
    ap.add_argument("--skip-backward-steps", type=int, default=1)
    # quality guidance (Kontext FreqEdit only)
    ap.add_argument("--qg-time-scale", type=float, default=None)
    ap.add_argument("--qg-lambda", type=float, default=None)
    ap.add_argument("--prompt-preserve", default="A high quality picture.")

    # memory
    ap.add_argument("--offload", choices=["auto", "on", "off"], default="auto",
                    help="CPU offload; auto = on for Qwen, off for Kontext")
    return ap.parse_args()


def load_prompts(args):
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        from example_data.param_prompts import prompts as all_prompts
        prompts = all_prompts[args.prompt_idx - 1]
    if len(prompts) < args.turns:
        raise ValueError(f"only {len(prompts)} prompts for {args.turns} turns")
    return prompts[: args.turns]


def build_pipe(args):
    if args.model == "kontext":
        if args.method == "native":
            from diffusers import FluxKontextPipeline as Pipe
        else:
            from pipelines.FreqEditKontext_pipeline import FreqEditKontextPipeline as Pipe
    else:
        if args.method == "native":
            from diffusers import QwenImageEditPipeline as Pipe
        else:
            from pipelines.FreqEditQwen_pipeline import FreqEditQwenEditPipeline as Pipe
    pipe = Pipe.from_pretrained(MODEL_IDS[args.model], torch_dtype=torch.bfloat16)
    offload = args.offload == "on" or (args.offload == "auto" and args.model == "qwen")
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    return pipe, offload


def freqedit_kwargs(args):
    d = DEFAULTS[args.model]
    pick = lambda name: getattr(args, name) if getattr(args, name) is not None else d[name]
    kw = dict(
        Inject_time_scale=pick("inject_time_scale"),
        base_alpha=pick("base_alpha"),
        gamma=pick("gamma"),
        levels=args.levels,
        wavelet_mode=args.wavelet,
        use_adaptive_weight=not args.no_adaptive_weight,
        backward_steps=args.backward_steps,
        skip_backward_steps=args.skip_backward_steps,
    )
    if args.model == "kontext":
        kw.update(QG_time_scale=pick("qg_time_scale"), QG_lambda=pick("qg_lambda"),
                  prompt_preserve=args.prompt_preserve)
    return kw


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "diffusers", "transformers", "pytorch_wavelets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        out["gpu"] = None
    return out


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=here, text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def main():
    args = parse_args()
    prompts = load_prompts(args)

    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    run = f"{args.model}_{args.method}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out_root, name, run)
    os.makedirs(out_dir, exist_ok=True)
    first_new = os.path.join(out_dir, f"turn_{args.start_turn + 1:02d}.png")
    if os.path.exists(first_new) and not args.overwrite:
        raise FileExistsError(f"{first_new} exists; use --overwrite or a different --tag")

    size = (args.size, args.size)
    source = load_image(args.image).convert("RGB").resize(size)
    if args.start_turn == 0:
        source.save(os.path.join(out_dir, "turn_00.png"))
        current = source
    else:
        current = load_image(os.path.join(out_dir, f"turn_{args.start_turn:02d}.png")).convert("RGB").resize(size)

    pipe, offload = build_pipe(args)
    fe_kw = freqedit_kwargs(args) if args.method == "freqedit" else {}

    # every argument that reaches pipe(...), apart from prompt / image / generator, which change per turn
    call_kwargs = dict(height=args.size, width=args.size, num_inference_steps=args.steps)
    if args.model == "kontext":
        call_kwargs["guidance_scale"] = args.guidance
    call_kwargs.update(fe_kw)

    params = dict(
        model=args.model, model_id=MODEL_IDS[args.model], method=args.method,
        pipeline_class=type(pipe).__name__,
        source_image=os.path.abspath(args.image),
        prompts_from=os.path.abspath(args.prompt_file) if args.prompt_file else f"param_prompts.prompts{args.prompt_idx}",
        turns=args.turns, seed_rule="turn k uses seed + k", seed=args.seed,
        source_image_for_quality_guidance=(args.model == "kontext" and args.method == "freqedit"),
        pipe_call_kwargs=call_kwargs,
        cpu_offload=offload,
        versions=versions(),
        repo_commit=git_commit(),
        command="python " + " ".join(sys.argv),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    params_path = os.path.join(out_dir, "params.json")
    if args.start_turn == 0 or not os.path.exists(params_path):
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
    print(json.dumps(params["pipe_call_kwargs"], indent=2))

    log = dict(prompts=prompts, seeds={k: args.seed + k for k in range(1, args.turns + 1)}, turn_seconds={})
    cfg_path = os.path.join(out_dir, "log.json")
    if args.start_turn > 0 and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            log["turn_seconds"] = json.load(f).get("turn_seconds", {})

    for k in range(args.start_turn + 1, args.turns + 1):
        prompt = prompts[k - 1]
        print(f"\n[{run}] turn {k}/{args.turns}: {prompt}")
        # exactly the kwargs recorded in params.json, plus the per-turn ones
        kw = dict(call_kwargs, prompt=prompt, image=current,
                  generator=torch.Generator(device="cuda").manual_seed(args.seed + k))
        if params["source_image_for_quality_guidance"]:
            kw["source_image"] = source

        t0 = time.time()
        current = pipe(**kw).images[0]
        log["turn_seconds"][k] = round(time.time() - t0, 1)

        current.save(os.path.join(out_dir, f"turn_{k:02d}.png"))
        with open(cfg_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"saved turn_{k:02d}.png ({log['turn_seconds'][k]} s)")

    print(f"\ndone: {out_dir}")


if __name__ == "__main__":
    main()