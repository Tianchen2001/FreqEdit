"""Model-independent part of FreqEdit, shared by FreqEditKontext_pipeline.py and FreqEditQwen_pipeline.py.

The two pipelines keep their own denoising loops (transformer calls, latent packing, quality guidance);
everything that decides *what* FreqEdit does at a step lives here:

    FreqEditConfig      all knobs (injection window, wavelet bands, alpha, path compensation, QG window)
    blend_bands()       generalization of utils.correct_with_wavelet_guidance: choose which DWT components
                        are pulled toward the reference velocity
    FreqEditController  per-call state: which steps inject / run QG, the corrected velocity,
                        and the accumulated path compensation

With the default config (bands = all detail levels, orient = "all", no alpha cap, window starting at 0,
compensation on) the result is identical to the released FreqEdit code.

DWT naming: a J-level DWT of the velocity gives [LL_J, D_J, ..., D_1]; D_1 is the finest detail level.
Each D_j holds three orientations (LH, HL, HH). The released FreqEdit uses J = 2 and pulls D_1 and D_2,
keeping LL_2 from the model.
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from utils import adaptive_weight, build_image_from_dwt_pyramid, dwt_pyramid

ORIENT_INDEX = {"all": (0, 1, 2), "LHHL": (0, 1), "HH": (2,)}  # pytorch_wavelets order: LH, HL, HH


@dataclass
class FreqEditConfig:
    # high-frequency injection, steps [int(inject_start * N), int(inject_end * N))
    inject_start: float = 0.0
    inject_end: float = 0.3                 # "Inject_time_scale" in the released code
    levels: int = 2                         # J
    wavelet: str = "db4"
    inject_bands: Optional[Tuple[str, ...]] = None   # subset of ("LL", "D1", ..., "DJ"); None = all detail levels
    orient: str = "all"                     # "all" | "HH" | "LHHL": orientations of D_j that are pulled
    use_adaptive_weight: bool = True
    base_alpha: float = 1.6
    gamma: float = 2.0
    alpha_max: Optional[float] = None       # cap on the alpha map (None = no cap, as released)
    # path compensation
    compensation: bool = True
    backward_steps: int = 4
    skip_backward_steps: int = 1            # counted from the start of the injection window
    # quality guidance (Kontext only), steps [int(qg_start * N), int(qg_end * N)), never on an injection step
    qg_start: float = 0.7                   # "QG_time_scale" in the released code
    qg_end: float = 1.0
    qg_lambda: float = 0.0

    def bands(self):
        if self.inject_bands is None:
            return tuple(f"D{j}" for j in range(1, self.levels + 1))
        bands = tuple(self.inject_bands)
        valid = {"LL"} | {f"D{j}" for j in range(1, self.levels + 1)}
        bad = [b for b in bands if b not in valid]
        if bad:
            raise ValueError(f"inject_bands {bad} not valid for levels={self.levels}; choose from {sorted(valid)}")
        return bands

    def to_dict(self):
        d = asdict(self)
        d["inject_bands"] = list(self.bands())
        return d


def blend_bands(v_pred, v_ref, alpha_map, wavelet="db4", levels=2, bands=None, orient="all"):
    """(1 - alpha) * pred + alpha * ref on the chosen DWT components; the others keep v_pred.

    bands=None and orient="all" reproduce utils.correct_with_wavelet_guidance exactly, including how the
    alpha map is resized (coarsest detail level first, then each finer level from the previous size).
    """
    if bands is None:
        bands = tuple(f"D{j}" for j in range(1, levels + 1))
    keep_or = ORIENT_INDEX[orient]

    dtype = v_pred.dtype
    v_pred = v_pred.to(torch.float32)
    v_ref = v_ref.to(torch.float32)

    pred_pyramid = dwt_pyramid(v_pred, levels, wavelet)
    keep_pyramid = dwt_pyramid(v_ref, levels, wavelet)

    # alpha for each pyramid entry, resized in the same chain as the released code
    alphas = []
    a = alpha_map
    for i in range(1, levels + 1):
        a = F.interpolate(a, size=pred_pyramid[i].shape[-2:], mode="bilinear", align_corners=False)
        alphas.append(a)

    corrected_pyramid = []
    if "LL" in bands:  # LL_J has the size of D_J, so it uses the first alpha of the chain
        corrected_pyramid.append((1 - alphas[0]) * pred_pyramid[0] + alphas[0] * keep_pyramid[0])
    else:
        corrected_pyramid.append(pred_pyramid[0])

    for i in range(1, levels + 1):
        p_pred, p_keep, a = pred_pyramid[i], keep_pyramid[i], alphas[i - 1]
        if f"D{levels + 1 - i}" not in bands:  # pyramid index i is detail level J + 1 - i
            corrected_pyramid.append(p_pred)
            continue
        p_corrected = (1 - a) * p_pred + a * p_keep
        if len(keep_or) < 3:
            drop = [k for k in range(3) if k not in keep_or]
            p_corrected[:, :, drop] = p_pred[:, :, drop]
        corrected_pyramid.append(p_corrected)

    v_corrected = build_image_from_dwt_pyramid(corrected_pyramid, wavelet)
    return v_corrected.to(dtype=dtype)


class FreqEditController:
    """Schedule and state of FreqEdit for one pipeline call."""

    def __init__(self, config: FreqEditConfig, num_inference_steps: int, sigmas, latents):
        self.cfg = config
        self.sigmas = sigmas
        n = num_inference_steps
        self.inject_start = int(config.inject_start * n)
        self.inject_end = int(config.inject_end * n)
        self.qg_start = int(config.qg_start * n)
        self.qg_end = int(config.qg_end * n)
        self.bands = config.bands()
        self.backward_v = torch.zeros_like(latents)
        self.accumulated_cnt = 0

    # ---------------------------------------------------------------- schedule
    def inject(self, i):
        return self.inject_start <= i < self.inject_end

    def quality_guidance(self, i):
        # the released code uses `elif`: QG never runs on an injection step
        return self.cfg.qg_lambda > 0 and self.qg_start <= i < self.qg_end and not self.inject(i)

    # ---------------------------------------------------------------- injection
    def reference_velocity(self, i, latents, ref_latents):
        """Velocity that takes the current latents straight to ref_latents at the last sigma."""
        return (ref_latents - latents) / (self.sigmas[-1] - self.sigmas[i])

    def alpha_map(self, v_pred, v_ref):
        if self.cfg.use_adaptive_weight:
            a = adaptive_weight(v_ref, v_pred, self.cfg.base_alpha, gamma=self.cfg.gamma)
        else:
            a = torch.full((v_ref.shape[0], 1, *v_ref.shape[-2:]), self.cfg.base_alpha,
                           device=v_ref.device, dtype=v_ref.dtype)
        if self.cfg.alpha_max is not None:
            a = a.clamp(max=self.cfg.alpha_max)
        return a

    def correct(self, v_pred, v_ref):
        """v_pred, v_ref: unpacked 4D velocities (B, C, H, W)."""
        return blend_bands(v_pred, v_ref, self.alpha_map(v_pred, v_ref), wavelet=self.cfg.wavelet,
                           levels=self.cfg.levels, bands=self.bands, orient=self.cfg.orient)

    # ---------------------------------------------------------------- path compensation
    def accumulate(self, i, noise_pred, corrected_noise_pred):
        if self.cfg.compensation and i - self.inject_start > self.cfg.skip_backward_steps:
            self.backward_v += (self.sigmas[i + 1] - self.sigmas[i]) * (noise_pred - corrected_noise_pred)
            self.accumulated_cnt += 1

    def post_step(self, i, latents):
        """Call right after scheduler.step; adds the accumulated compensation back every backward_steps."""
        if self.accumulated_cnt == self.cfg.backward_steps or i == self.inject_end - 1:
            latents += self.backward_v
            self.accumulated_cnt = 0
            self.backward_v = torch.zeros_like(latents)
        return latents
