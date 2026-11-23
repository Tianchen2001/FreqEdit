"""VLM judge for multi-turn editing chains, following the FreqEdit appendix B protocol
(itself adapted from EdiVal-Agent): per turn k, the judge sees X[k] (before), X[k+1] (after) and the
instruction p[k], and returns an instruction-following score in [0, 1], ignoring visual quality.

Optional --artifacts adds a second, separate call per turn that compares X[k+1] with X[k] and flags new
artifacts (noise, blur, saturation, ghosting, distortion). EdiVal-Agent (App. L) finds zero-shot VLMs
unreliable as quality scorers, so use these flags to find chains worth looking at, not as a reported metric.

Put in FreqEdit/src. Needs one of
    pip install openai        and  export OPENAI_API_KEY=...   (--backend openai, model gpt-4o: as in FreqEdit)
    pip install google-genai  and  export GEMINI_API_KEY=...   (--backend gemini, e.g. gemini-2.5-flash)

    python vlm_eval.py "output_nostyle/*/kontext_*" --summary vlm/nostyle_kontext.csv
    python vlm_eval.py "output_nostyle_A3/*/kontext_freqedit" --tag A3 --summary vlm/nostyle_A3.csv
    python vlm_eval.py "output_nostyle/*/kontext_*" --turns 1 5 10 --dry-run      # prints the plan and the cost

Per run folder it writes vlm.json (one entry per scored turn; re-runs only score what is missing) and
the --summary csv: one row per image, run and turn, with if_score, if_reason and the artifact flags.
"""
import argparse
import base64
import concurrent.futures as cf
import csv
import glob
import io
import json
import os
import re
import threading
import time

from PIL import Image

IF_PROMPT = """You are an objective evaluator of instruction-based image editing.
Image 1 is the image BEFORE the edit. Image 2 is the image AFTER the edit.
The editing instruction was:
"{instruction}"

Judge ONLY whether the requested change was carried out in Image 2 relative to Image 1: the right object, attribute or region changed in the requested way.
- Ignore image quality, sharpness, noise, color fidelity and artifacts; they are evaluated separately.
- Ignore whether unrelated content also changed; consistency is evaluated separately. A clause such as "while maintaining ...", "keeping ..." or "preserving ..." describes what should stay unchanged: do not score it.
- 1.0 means the requested change is fully and clearly done; 0.5 means partly done or ambiguous; 0.0 means not done or a different change was made. Intermediate values are allowed.

Respond with JSON only: {{"score": <number between 0 and 1>, "reason": "<one short sentence>"}}"""

ART_PROMPT = """You are an expert in image quality. Image 1 is an image BEFORE an edit, Image 2 is the same image AFTER the edit.
The intended edit was: "{instruction}". Do NOT judge whether the edit was done.
Compare Image 2 with Image 1 and report only problems that are NEW or WORSE in Image 2, outside the intentionally edited content.
For each item give 0 (none), 1 (slight) or 2 (clear):
- noise: grain, speckles or blotches, especially in flat regions (walls, sky, skin)
- blur: loss of fine detail or texture, smearing, plastic-looking surfaces
- color: oversaturation, contrast increase or color cast
- ghosting: double edges, halos, glow, semi-transparent leftovers
- distortion: deformed faces, hands, bodies or objects, changed proportions, cartoon-like look
- shift: the whole scene moved, zoomed or changed perspective

Respond with JSON only: {{"noise": n, "blur": n, "color": n, "ghosting": n, "distortion": n, "shift": n, "note": "<one short sentence>"}}"""

ART_KEYS = ["noise", "blur", "color", "ghosting", "distortion", "shift"]


# ---------------------------------------------------------------- backends
def png_bytes(path, side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class OpenAIJudge:
    def __init__(self, model):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def __call__(self, prompt, images):
        content = [{"type": "text", "text": prompt}]
        for b in images:
            url = "data:image/png;base64," + base64.b64encode(b).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}])
        u = r.usage
        return r.choices[0].message.content, (u.prompt_tokens, u.completion_tokens)


class GeminiJudge:
    def __init__(self, model):
        from google import genai
        from google.genai import types
        self.client, self.model, self.types = genai.Client(), model, types

    def __call__(self, prompt, images):
        t = self.types
        parts = [t.Part.from_bytes(data=b, mime_type="image/png") for b in images] + [prompt]
        cfg = dict(temperature=0, response_mime_type="application/json")
        if "flash" in self.model:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
        r = self.client.models.generate_content(model=self.model, contents=parts,
                                                config=t.GenerateContentConfig(**cfg))
        u = r.usage_metadata
        return r.text, (u.prompt_token_count or 0, (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


# ---------------------------------------------------------------- per run
def jobs_for_run(run_dir, turns, artifacts, done):
    log = os.path.join(run_dir, "log.json")
    prompts = json.load(open(log)).get("prompts", []) if os.path.exists(log) else []
    out = []
    for k in turns:
        a = os.path.join(run_dir, f"turn_{k - 1:02d}.png")
        b = os.path.join(run_dir, f"turn_{k:02d}.png")
        if not (os.path.exists(a) and os.path.exists(b)) or k > len(prompts):
            continue
        prev = done.get(str(k), {})
        if "if_score" not in prev:
            out.append((run_dir, k, "if", a, b, prompts[k - 1]))
        if artifacts and "noise" not in prev:
            out.append((run_dir, k, "art", a, b, prompts[k - 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, globs ok, e.g. 'output_nostyle/*/kontext_*'")
    ap.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    ap.add_argument("--model", default=None, help="default gpt-4o (openai) / gemini-2.5-flash (gemini)")
    ap.add_argument("--turns", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--artifacts", action="store_true", help="also ask for artifact flags (second call per turn)")
    ap.add_argument("--side", type=int, default=768, help="images are downscaled to this size before sending")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="appended to the run name in the summary (e.g. A3)")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--dry-run", action="store_true", help="count the calls and estimate cost, call nothing")
    args = ap.parse_args()
    model = args.model or ("gpt-4o" if args.backend == "openai" else "gemini-2.5-flash")

    run_dirs = sorted({d for p in args.runs for d in glob.glob(p) if os.path.isdir(d)})
    state = {}
    jobs = []
    for d in run_dirs:
        path = os.path.join(d, "vlm.json")
        state[d] = json.load(open(path)) if os.path.exists(path) else {"model": model, "turns": {}}
        if state[d].get("model") != model and state[d]["turns"]:
            print(f"note: {d}/vlm.json was scored with {state[d].get('model')}; new turns use {model}")
        jobs += jobs_for_run(d, args.turns, args.artifacts, state[d]["turns"])

    # rough cost: ~2.2k input tokens and ~80 output tokens per call (two 768px images)
    price = {"gpt-4o": (2.5, 10), "gemini-2.5-flash": (0.3, 2.5), "gemini-2.5-pro": (1.25, 10)}.get(model)
    est = f", about ${len(jobs) * (2200 * price[0] + 80 * price[1]) / 1e6:.2f}" if price else ""
    print(f"{len(run_dirs)} runs, {len(jobs)} calls to {model}{est}")
    if args.dry_run or not jobs:
        if args.dry_run:
            return
    else:
        judge = OpenAIJudge(model) if args.backend == "openai" else GeminiJudge(model)
        lock = threading.Lock()
        tokens = [0, 0]

        def run(job):
            d, k, kind, a, b, instr = job
            prompt = (IF_PROMPT if kind == "if" else ART_PROMPT).format(instruction=instr.replace('"', "'"))
            imgs = [png_bytes(a, args.side), png_bytes(b, args.side)]
            for attempt in range(5):
                try:
                    text, (ti, to) = judge(prompt, imgs)
                    r = parse_json(text)
                    if kind == "if":
                        res = {"if_score": max(0.0, min(1.0, float(r["score"]))), "if_reason": str(r.get("reason", ""))}
                    else:
                        res = {key: int(r.get(key, 0)) for key in ART_KEYS}
                        res["art_note"] = str(r.get("note", ""))
                    break
                except Exception as e:  # rate limits, bad JSON
                    if attempt == 4:
                        print(f"  failed {d} turn {k} {kind}: {e}")
                        return
                    time.sleep(2 ** attempt * 2)
            with lock:
                state[d]["turns"].setdefault(str(k), {}).update(res)
                tokens[0] += ti; tokens[1] += to
                with open(os.path.join(d, "vlm.json"), "w") as f:
                    json.dump(state[d], f, indent=1)

        with cf.ThreadPoolExecutor(args.workers) as ex:
            for i, _ in enumerate(ex.map(run, jobs), 1):
                if i % 50 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} calls, tokens in {tokens[0]} out {tokens[1]}")

    if args.summary:
        rows = []
        for d in run_dirs:
            image = os.path.basename(os.path.dirname(os.path.normpath(d)))
            run = os.path.basename(os.path.normpath(d)) + (f"_{args.tag}" if args.tag else "")
            for k, v in sorted(state[d]["turns"].items(), key=lambda kv: int(kv[0])):
                rows.append({"image": image, "run": run, "turn": int(k), **v})
        if rows:
            keys = ["image", "run", "turn", "if_score", "if_reason"] + ART_KEYS + ["art_note"]
            os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
            with open(args.summary, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"summary: {args.summary} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
