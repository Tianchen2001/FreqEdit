"""CPU checks for freqedit_core.py: the refactor must reproduce the released FreqEdit logic exactly.

Run from src/:  python pipelines/test_freqedit_core.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import adaptive_weight, correct_with_wavelet_guidance, dwt_pyramid  # noqa: E402
from pipelines.freqedit_core import FreqEditConfig, FreqEditController, blend_bands  # noqa: E402

torch.manual_seed(0)
B, C, H, W = 1, 16, 64, 64


def rand(dtype=torch.float32):
    return torch.randn(B, C, H, W).to(dtype)


def test_blend_default_is_released():
    for dtype in (torch.float32, torch.bfloat16):
        for levels in (1, 2):
            for adaptive in (True, False):
                vp, vr = rand(dtype), rand(dtype)
                a = adaptive_weight(vr, vp, 1.6, gamma=2.0) if adaptive else 1.6 * torch.ones_like(vp[:, :1])
                old = correct_with_wavelet_guidance(vp, vr, alpha_map=a, wavelet="db4", levels=levels)
                new = blend_bands(vp, vr, a, wavelet="db4", levels=levels)
                assert torch.equal(old, new), (dtype, levels, adaptive)


def test_blend_limits():
    vp, vr = rand(), rand()
    one = torch.ones(B, 1, H, W)
    full = blend_bands(vp, vr, one, levels=2, bands=("LL", "D1", "D2"))
    assert torch.allclose(full, vr, atol=1e-4), "all bands with alpha 1 should give v_ref"
    none = blend_bands(vp, vr, one, levels=2, bands=())
    assert torch.allclose(none, vp, atol=1e-4), "no bands should give v_pred"


def test_blend_linear_in_bands_and_orient():
    vp, vr = rand(), rand()
    a = adaptive_weight(vr, vp, 1.6, gamma=2.0)
    f = lambda **k: blend_bands(vp, vr, a, levels=2, **k)
    base = f(bands=())
    # D1 + D2 = default, LL + D1 + D2 = all, HH + LHHL = all orientations
    assert torch.allclose(f(bands=("D1",)) + f(bands=("D2",)) - base, f(), atol=1e-4)
    assert torch.allclose(f(bands=("LL",)) + f() - base, f(bands=("LL", "D1", "D2")), atol=1e-4)
    assert torch.allclose(f(orient="HH") + f(orient="LHHL") - base, f(), atol=1e-4)


def test_orient_index_is_diagonal():
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = ((x + y) % 2).float()[None, None]                 # pure diagonal structure
    d1 = dwt_pyramid(checker, 2, "db1")[2]                       # finest detail level, (B, C, 3, h, w)
    e = d1.pow(2).sum(dim=(0, 1, 3, 4))
    assert e.argmax().item() == 2, f"HH should be index 2, energies {e.tolist()}"


def test_bad_band():
    try:
        FreqEditConfig(levels=1, inject_bands=("D2",)).bands()
    except ValueError:
        return
    raise AssertionError("D2 with levels=1 should be rejected")


# ---------------------------------------------------------------- whole loop vs the released loop
def released_loop(vs, z0, zref, sigmas, n, Inject_time_scale, base_alpha, gamma, levels, backward_steps,
                  skip_backward_steps, QG_time_scale, QG_lambda):
    """The denoising-loop logic of FreqEditKontext_pipeline.py before the refactor (packing = identity,
    scheduler.step = Euler, QG replaced by a fixed pull so that it is visible)."""
    Inject_timesteps = int(Inject_time_scale * n)
    QG_timesteps = int(QG_time_scale * n)
    backward_v = torch.zeros_like(z0)
    accumulated_cnt = 0
    latents = z0.clone()
    for i in range(n):
        noise_pred = vs[i].clone()
        if i < Inject_timesteps:
            ref_image_noise_pred = (zref - latents) / (sigmas[-1] - sigmas[i])
            alpha_map = adaptive_weight(ref_image_noise_pred, noise_pred, base_alpha, gamma=gamma)
            corrected_noise_pred = correct_with_wavelet_guidance(noise_pred, ref_image_noise_pred,
                                                                 alpha_map=alpha_map, wavelet="db4", levels=levels)
            if i > skip_backward_steps:
                backward_v += (sigmas[i + 1] - sigmas[i]) * (noise_pred - corrected_noise_pred)
                accumulated_cnt += 1
            noise_pred = corrected_noise_pred
        elif QG_lambda > 0 and i >= QG_timesteps:
            noise_pred = noise_pred + QG_lambda * (vs[n] - noise_pred)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * noise_pred
        if accumulated_cnt == backward_steps or i == Inject_timesteps - 1:
            latents += backward_v
            accumulated_cnt = 0
            backward_v = torch.zeros_like(latents)
    return latents


def refactored_loop(vs, z0, zref, sigmas, n, cfg):
    latents = z0.clone()
    controller = FreqEditController(cfg, n, sigmas, latents)
    for i in range(n):
        noise_pred = vs[i].clone()
        if controller.inject(i):
            ref_image_noise_pred = controller.reference_velocity(i, latents, zref)
            corrected_noise_pred = controller.correct(noise_pred, ref_image_noise_pred)
            controller.accumulate(i, noise_pred, corrected_noise_pred)
            noise_pred = corrected_noise_pred
        elif controller.quality_guidance(i):
            noise_pred = noise_pred + cfg.qg_lambda * (vs[n] - noise_pred)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * noise_pred
        latents = controller.post_step(i, latents)
    return latents


def test_loop_matches_released():
    n = 28
    sigmas = torch.linspace(1, 0, n + 1)
    vs = [rand() for _ in range(n + 1)]
    z0, zref = rand(), rand()
    for p in [dict(Inject_time_scale=0.3, base_alpha=1.6, gamma=2.0, levels=2, QG_time_scale=0.7, QG_lambda=0.3),
              dict(Inject_time_scale=0.3, base_alpha=2.0, gamma=1.6, levels=2, QG_time_scale=0.7, QG_lambda=0.0),
              dict(Inject_time_scale=0.5, base_alpha=5.0, gamma=1.0, levels=1, QG_time_scale=0.7, QG_lambda=0.5)]:
        old = released_loop(vs, z0, zref, sigmas, n, backward_steps=4, skip_backward_steps=1, **p)
        cfg = FreqEditConfig(inject_end=p["Inject_time_scale"], base_alpha=p["base_alpha"], gamma=p["gamma"],
                             levels=p["levels"], qg_start=p["QG_time_scale"], qg_lambda=p["QG_lambda"])
        new = refactored_loop(vs, z0, zref, sigmas, n, cfg)
        assert torch.equal(old, new), p


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
