"""
Two-Stage Golden-Cell Evaluation
==================================
Stage 1: Run K row + K col passes on K² frames (cheap, low-res if desired).
         Build importance map M[r,c] = row_conf[r] * col_conf[c].
         Select top-M golden cells.

Stage 2: Run ONE baseline pass on just the M golden-cell frames at FULL resolution.
         Score letters → final answer.

Comparison: also runs full baseline at K² frames for reference.

Usage:
    python -m GridProbe.eval.two_stage_eval \
        --data_dir /path/to/Video-MME \
        --K 8 --M 8 --n_samples 30 --debug
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, score_letters_for_pass,
    get_letter_token_id_variants, row_indices, col_indices,
    load_video_mme, MERGE_FNS,
)
from GridProbe.eval.video_mme import build_prompt as build_prompt_v1, check_answer as check_answer_v1
from GridProbe.eval.video_mme_v2 import (
    LETTERS_V2, build_prompt_v2, check_answer_v2, load_video_mme_v2, BINS_V2,
)
from GridProbe.eval.longvideobench import (
    LETTERS_LVB, BINS_LVB, build_prompt_lvb, check_answer_lvb, load_lvb,
)

logger = logging.getLogger(__name__)

# Module-level constants — reassigned by main() based on --benchmark.
LETTERS = ["A", "B", "C", "D"]
build_prompt = build_prompt_v1
check_answer = check_answer_v1


# ═══════════════════════════════════════════════════════════════
# Stage 1: cheap grid probe → importance map → golden cells
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def stage1_probe(model, processor, prompt_text, all_frames, K,
                 letter_variants, device, max_video_pixels_probe):
    """Run K row + K col passes. Returns importance map + per-axis info.

    Returns dict with:
        row_confs, col_confs: (K,) arrays
        row_preds, col_preds: list[str]
        row_lp, col_lp: (K, 4) tensors
        M_mult: (K, K) importance map
        golden_indices: list of (r, c) sorted by descending importance
    """
    # ── Row passes ──
    row_lp_list = []
    for indices in row_indices(K):
        subset = [all_frames[i] for i in indices]
        lp = score_letters_for_pass(
            model, processor, prompt_text, subset,
            letter_variants, device, max_video_pixels_probe)
        row_lp_list.append(lp.cpu())
    row_lp = torch.stack(row_lp_list)

    # ── Col passes ──
    col_lp_list = []
    for indices in col_indices(K):
        subset = [all_frames[i] for i in indices]
        lp = score_letters_for_pass(
            model, processor, prompt_text, subset,
            letter_variants, device, max_video_pixels_probe)
        col_lp_list.append(lp.cpu())
    col_lp = torch.stack(col_lp_list)

    # ── Confidences ──
    row_probs = F.softmax(row_lp, dim=-1)
    col_probs = F.softmax(col_lp, dim=-1)
    row_confs = row_probs.max(dim=-1).values.numpy()
    col_confs = col_probs.max(dim=-1).values.numpy()
    row_preds = [LETTERS[i] for i in row_lp.argmax(dim=-1).tolist()]
    col_preds = [LETTERS[i] for i in col_lp.argmax(dim=-1).tolist()]

    # ── Importance map ──
    M_mult = row_confs[:, None] * col_confs[None, :]  # (K, K)

    # ── Rank all cells by importance ──
    flat = M_mult.flatten()
    sorted_idx = np.argsort(flat)[::-1]
    golden = [(int(idx // K), int(idx % K)) for idx in sorted_idx]

    # ── Ensemble prediction from probe passes ──
    all_lp = torch.cat([row_lp, col_lp], dim=0)
    ensemble_pred = LETTERS[all_lp.mean(dim=0).argmax().item()]

    return {
        "row_confs": row_confs, "col_confs": col_confs,
        "row_preds": row_preds, "col_preds": col_preds,
        "row_lp": row_lp, "col_lp": col_lp,
        "M_mult": M_mult,
        "golden": golden,
        "ensemble_pred": ensemble_pred,
    }


# ═══════════════════════════════════════════════════════════════
# Adaptive M selection via Participation Ratio
# ═══════════════════════════════════════════════════════════════

def participation_ratio(M_mult, subtract_min: bool = True, gamma: float = 3.0,
                         min_M: int = 1, max_M: int = None):
    """Compute adaptive number of frames to keep via participation ratio.

    PR = 1 / sum(p_i^2)  where p_i is normalized importance.

    In narrow-confidence regimes (e.g. [0.5, 0.95]), raw importances are too
    uniform → PR inflates → M too large. We apply gamma sharpening:
        p_i ∝ (M_i - min)^gamma
    gamma=1: raw participation ratio (returns ~K² for uniform)
    gamma=2-3: balanced — rewards peaks without collapsing to 1
    gamma=5+: aggressive peak selection (M → few frames)

    Also uses cumulative-mass capping: if even after sharpening PR > (mass_cap
    threshold where cumulative mass hits 0.9), fall back to that smaller number.

    Args:
        M_mult: (K, K) importance map
        subtract_min: center on excess-over-baseline (recommended)
        gamma: sharpening exponent (>1 sharpens peaks)
        min_M: minimum frames to keep (safety floor)
        max_M: cap on frames (None = K²)

    Returns: int, number of frames to keep
    """
    flat = M_mult.flatten().astype(np.float64)
    if subtract_min:
        flat = flat - flat.min()
    # Sharpening: (|x|)^gamma, preserving sign handling for safety
    flat = np.clip(flat, 0, None) ** gamma
    total = flat.sum()
    if total <= 0:
        return min_M
    p = flat / total
    pr = 1.0 / (p * p).sum()

    # Cross-check with cumulative mass (pick smallest M capturing 90% of sharpened mass)
    sorted_p = np.sort(p)[::-1]
    cum = np.cumsum(sorted_p)
    mass_M = int(np.searchsorted(cum, 0.90) + 1)

    # Use the TIGHTER of the two estimates
    M = min(int(np.ceil(pr)), mass_M)
    K2 = len(flat)
    hi = max_M if max_M is not None else K2
    return int(np.clip(M, min_M, hi))


# ─── Alternative selectors (for --selector flag) ────────────────

def mean_threshold_selector(M_mult, min_M: int = 1, max_M: int = None):
    """Pick all cells strictly above the mean importance.

    Intuition (user's idea): skewed distributions → mean pulled up by peaks →
    fewer cells pass the threshold. Uniform-ish distributions → ~50% pass.
    """
    flat = M_mult.flatten().astype(np.float64)
    mean_val = flat.mean()
    count = int((flat > mean_val).sum())
    K2 = len(flat)
    hi = max_M if max_M is not None else K2
    return int(np.clip(count, min_M, hi))


def skewness_selector(M_mult, min_M: int = 1, max_M: int = None, scale: float = 3.0):
    """Scale M inversely with |skewness|. Asymmetric (skewed OR reverse-skewed) → fewer frames.

    M = K² / (1 + scale · |skew|)

    Our importance maps often have slight negative skew (confidences clustered
    high with a few low outliers), which is still informative — it means few
    frames are "different from the baseline high-confidence crowd". Both
    signs of skew carry signal, so we use |skew|.

    - |skew|=0 (uniform):        M = K²
    - |skew|=1  at scale=3:      M = K²/4  → 16 for K=8
    - |skew|=2  at scale=3:      M = K²/7  → 9  for K=8
    - |skew|=3  at scale=3:      M = K²/10 → 6  for K=8

    Excess kurtosis (peakedness) is the natural companion — add it if skew
    alone saturates on uniform-ish maps.
    """
    flat = M_mult.flatten().astype(np.float64)
    mu = flat.mean()
    sd = flat.std()
    K2 = len(flat)
    if sd < 1e-10:
        return min(K2, max_M or K2)
    # Distribution-shape selector: combines two complementary moments.
    #   skewness  (3rd) — captures asymmetric concentration of evidence
    #   kurt-excess (4th) — captures peakedness independent of asymmetry
    # Either signal alone triggers some M reduction; together they respond
    # robustly across diverse importance-map shapes. Empirically:
    #   - pure skew misses peaked-but-symmetric maps
    #   - pure kurt fails on near-uniform maps (kurt_excess ≈ 0 → no reduction)
    skew = float(((flat - mu) ** 3).mean() / (sd ** 3))
    kurt_excess = float(((flat - mu) ** 4).mean() / (sd ** 4) - 3.0)
    shape_strength = abs(skew) + 0.5 * max(0.0, kurt_excess)
    denom = 1.0 + scale * shape_strength
    M = int(np.ceil(K2 / denom))
    hi = max_M if max_M is not None else K2
    return int(np.clip(M, min_M, hi))


def otsu_selector(M_mult, min_M: int = 1, max_M: int = None, n_bins: int = 64):
    """Otsu's method: find threshold that maximizes between-class variance.

    Classic image-segmentation recipe — splits a 1D distribution into two
    classes (important / unimportant) parameter-free. Count of cells in the
    "important" class is our M.
    """
    flat = M_mult.flatten().astype(np.float64)
    K2 = len(flat)
    if flat.std() < 1e-10:
        return min(K2, max_M or K2)

    # Histogram
    hist, edges = np.histogram(flat, bins=n_bins)
    total = hist.sum()
    if total == 0:
        return min_M

    # Cumulative sums for Otsu computation
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    cumsum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)
    global_mean = cum_mean[-1] / total

    # Between-class variance at each possible threshold bin
    best_bcv = -1.0
    best_thresh = bin_centers[0]
    for i in range(1, n_bins):
        w0 = cumsum[i - 1] / total
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[i - 1] / cumsum[i - 1]
        mu1 = (cum_mean[-1] - cum_mean[i - 1]) / (total - cumsum[i - 1])
        bcv = w0 * w1 * (mu0 - mu1) ** 2
        if bcv > best_bcv:
            best_bcv = bcv
            best_thresh = edges[i]

    count = int((flat > best_thresh).sum())
    hi = max_M if max_M is not None else K2
    return int(np.clip(count, min_M, hi))


def select_M(selector: str, M_mult, args):
    """Dispatcher. Returns adaptive M according to the chosen selector."""
    max_M = args.pr_max_M
    min_M = args.pr_min_M
    if selector == "pr":
        return participation_ratio(
            M_mult, subtract_min=args.pr_subtract_min,
            gamma=args.pr_gamma, min_M=min_M, max_M=max_M)
    elif selector == "mean":
        return mean_threshold_selector(M_mult, min_M=min_M, max_M=max_M)
    elif selector == "skew":
        return skewness_selector(M_mult, min_M=min_M, max_M=max_M)
    elif selector == "otsu":
        return otsu_selector(M_mult, min_M=min_M, max_M=max_M)
    else:
        raise ValueError(f"Unknown selector: {selector}")


# ─── Probe-pass cache (skip stage1 + baselines on re-runs) ──────

def _cache_path(cache_dir, sample, K, probe_px):
    """Unique cache filename per (question, K, probe resolution)."""
    qid = (sample.get("id") or sample.get("question_id")
           or hashlib.md5(sample["video_path"].encode()).hexdigest()[:12])
    safe_qid = str(qid).replace("/", "_").replace(" ", "_")
    return os.path.join(cache_dir, f"{safe_qid}_K{K}_p{probe_px}.pt")


def load_shared_cache(cache_dir, sample, K, probe_px):
    """Return cached probe+baselines dict, or None if missing."""
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, sample, K, probe_px)
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("  cache load failed for %s: %s", path, e)
        return None


def save_shared_cache(cache_dir, sample, K, probe_px, payload):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, sample, K, probe_px)
    try:
        torch.save(payload, path)
    except Exception as e:
        logger.warning("  cache save failed for %s: %s", path, e)


# ═══════════════════════════════════════════════════════════════
# Stage 2: focused pass on golden-cell frames
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def stage2_focused(model, processor, prompt_text, all_frames, golden_cells, M,
                   K, letter_variants, device, max_video_pixels_focus):
    """Run ONE focused pass on the top-M golden-cell frames at full resolution.

    NOTE: golden_cells is sorted by *descending importance* — but the VLM relies
    on temporal positional encoding, so we re-sort the chosen frames into
    chronological order before passing them to the model. Keeping importance
    order would scramble temporal context and break temporal-reasoning queries.
    """
    top_M = golden_cells[:M]
    # Convert (row, col) → flat frame index, drop OOB, then sort temporally.
    frame_indices = sorted([r * K + c for r, c in top_M
                             if (r * K + c) < len(all_frames)])
    selected_frames = [all_frames[idx] for idx in frame_indices]

    lp = score_letters_for_pass(
        model, processor, prompt_text, selected_frames,
        letter_variants, device, max_video_pixels_focus)

    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()

    return {
        "pred": pred,
        "conf": conf,
        "frame_indices": frame_indices,
        "lp": lp.cpu(),
    }


# ═══════════════════════════════════════════════════════════════
# Baselines for comparison
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_baseline_full(model, processor, prompt_text, all_frames,
                      letter_variants, device, max_video_pixels):
    """Standard baseline: one pass on all K² frames."""
    lp = score_letters_for_pass(
        model, processor, prompt_text, all_frames,
        letter_variants, device, max_video_pixels)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()
    return {"pred": pred, "conf": conf, "lp": lp.cpu()}


@torch.no_grad()
def run_baseline_random(model, processor, prompt_text, all_frames, M,
                        letter_variants, device, max_video_pixels):
    """Random-M baseline: one pass on M randomly selected frames."""
    idx = np.sort(np.random.choice(len(all_frames), size=min(M, len(all_frames)), replace=False))
    selected = [all_frames[i] for i in idx]
    lp = score_letters_for_pass(
        model, processor, prompt_text, selected,
        letter_variants, device, max_video_pixels)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()
    return {"pred": pred, "conf": conf, "frame_indices": idx.tolist(), "lp": lp.cpu()}


@torch.no_grad()
def run_baseline_uniform(model, processor, prompt_text, all_frames, M,
                         letter_variants, device, max_video_pixels):
    """Uniform-M baseline: one pass on M uniformly-spaced frames."""
    idx = np.linspace(0, len(all_frames) - 1, M).round().astype(int).tolist()
    selected = [all_frames[i] for i in idx]
    lp = score_letters_for_pass(
        model, processor, prompt_text, selected,
        letter_variants, device, max_video_pixels)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()
    return {"pred": pred, "conf": conf, "frame_indices": idx, "lp": lp.cpu()}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=False, default=None,
                   help="Required for video_mme/video_mme_v2. For lvb, see --lvb_json.")
    p.add_argument("--benchmark", choices=["video_mme", "video_mme_v2", "lvb"],
                   default="video_mme",
                   help="video_mme = 4-option (A-D); video_mme_v2 = 8-option (A-H); "
                        "lvb = LongVideoBench (4-option A-D, with subtitles by default).")
    p.add_argument("--lvb_json", default=None,
                   help="(lvb only) path to lvb_val.json (or to a stratified subset "
                        "produced by lvb_stratified_sample.py). The directory containing "
                        "this JSON is treated as the LVB root for resolving videos/ and "
                        "subtitles/ unless --lvb_dir is set.")
    p.add_argument("--lvb_dir", default=None,
                   help="(lvb only) optional override for LVB root dir (where videos/ "
                        "and subtitles/ live). Defaults to the parent of --lvb_json.")
    p.add_argument("--with_subtitle", action="store_true",
                   help="Include subtitles in the prompt. "
                        "v2: expects <data_dir>/subtitles/<vid>.jsonl. "
                        "lvb: reads subtitle_path from each LVB item.")
    p.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--K", type=int, default=8, help="Grid side (K²=total frames)")
    p.add_argument("--M", default="8",
                   help="Golden cells for stage 2. Integer (fixed), or 'auto' for "
                        "adaptive selection per the --selector rule.")
    p.add_argument("--selector", default="pr", choices=["pr", "mean", "skew", "otsu"],
                   help="Adaptive-M rule when --M=auto. "
                        "pr=participation ratio (sharpened), "
                        "mean=count cells above mean importance, "
                        "skew=scale K² by skewness, "
                        "otsu=bimodal split via Otsu's method.")
    p.add_argument("--pr_subtract_min", action="store_true", default=True,
                   help="Subtract min importance before computing PR (recommended)")
    p.add_argument("--pr_gamma", type=float, default=3.0,
                   help="Sharpening exponent for PR. 1=raw, 2-3=balanced, 5+=aggressive")
    p.add_argument("--pr_min_M", type=int, default=2,
                   help="Floor on adaptive M (avoid picking 1 frame only)")
    p.add_argument("--pr_max_M", type=int, default=None,
                   help="Ceiling on adaptive M (default K²)")
    p.add_argument("--max_video_pixels_probe", type=int, default=0,
                   help="Resolution for stage 1 probe passes (0=default, lower=faster)")
    p.add_argument("--max_video_pixels_focus", type=int, default=0,
                   help="Resolution for stage 2 focused pass (0=default=full res)")
    p.add_argument("--n_samples", type=int, default=30,
                   help="Cap on samples after sharding (≤0 = all samples in this shard).")
    p.add_argument("--duration_bin", default=None)
    p.add_argument("--n_shards", type=int, default=1,
                   help="Total number of parallel shards. Each shard sees samples[idx::n_shards].")
    p.add_argument("--shard_index", type=int, default=0,
                   help="0-based shard id (must be < n_shards).")
    p.add_argument("--parallel_gpus", type=int, default=1,
                   help="If > 1, spawn that many child processes (one per GPU) "
                        "with auto-sharding and merge results into --output. "
                        "Single command, no shell loop. Set to torch.cuda.device_count() "
                        "for full-machine parallelism.")
    p.add_argument("--parallel_stagger_sec", type=int, default=0,
                   help="Seconds to sleep between spawning each parallel worker. "
                        "Mitigates OOM during simultaneous model load. "
                        "Recommended 30-60 on memory-tight machines.")
    p.add_argument("--parallel_omp_threads", type=int, default=2,
                   help="OMP/MKL thread cap per parallel worker. Lower values "
                        "reduce RSS per child. Default 2 keeps 8 children comfortable "
                        "on most workstations.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default="two_stage_eval.json")
    p.add_argument("--probe_cache_dir", default=None,
                   help="Directory to cache probe passes + baseline preds. "
                        "First run writes; subsequent runs (e.g. different --selector) "
                        "skip stage1 and baselines for cached samples. "
                        "Cache keyed by (question_id, K, probe pixels).")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip_full_baseline", action="store_true",
                   help="Skip the K²-frame baseline forward (save ~28s/sample at K=16). "
                        "Useful when iterating on selector/PR knobs and the baseline is "
                        "already known / cached from a previous run.")
    p.add_argument("--skip_matched_baselines", action="store_true",
                   help="Skip the uniform-M and random-M matched-compute baselines. "
                        "Saves ~2*M_focus_pass per sample. Use when filling a fixed-M "
                        "row in the headline table and the matched-compute sanity row "
                        "isn't needed. base_uniform_M / base_random_M predictions "
                        "will be '(skip)' and their FLOPs/timings empty.")
    p.add_argument("--checkpoint_every", type=int, default=50,
                   help="Write partial JSON to --output every N samples so a crash "
                        "can be resumed. 0 disables checkpointing. Default 50.")
    p.add_argument("--no_resume", action="store_true",
                   help="If --output already exists, ignore it and start fresh. "
                        "Default is to resume: load existing results, skip those "
                        "sample IDs, and append new work to the same file.")
    args = p.parse_args()

    # ── Real-parallel: if --parallel_gpus N, fan out to N children and exit. ──
    from GridProbe.eval.parallel_launcher import maybe_spawn_parallel_workers
    if maybe_spawn_parallel_workers(args, eval_module="GridProbe.eval.two_stage_eval"):
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    K = args.K
    n_frames = K * K

    # ── Benchmark dispatch (V1 4-letter / V2 8-letter / LVB 4-5-letter) ──
    global LETTERS, build_prompt, check_answer
    if args.benchmark == "video_mme_v2":
        LETTERS = LETTERS_V2
        build_prompt = build_prompt_v2
        check_answer = check_answer_v2
        logger.info("Benchmark: Video-MME-v2 (8-option A-H)")
    elif args.benchmark == "lvb":
        LETTERS = LETTERS_LVB
        build_prompt = build_prompt_lvb
        check_answer = check_answer_lvb
        logger.info("Benchmark: LongVideoBench (5-option A-E max; per-Q candidates dynamic)")
        # Sanity-guard the required arg
        if not args.lvb_json:
            raise SystemExit("--benchmark lvb requires --lvb_json /path/to/lvb_val.json "
                             "(or to a stratified subset JSON).")
    else:
        LETTERS = ["A", "B", "C", "D"]
        build_prompt = build_prompt_v1
        check_answer = check_answer_v1
        logger.info("Benchmark: Video-MME (4-option A-D)")

    # ── Parse M: int or "auto" ──
    adaptive_M = (str(args.M).lower() == "auto")
    M_fixed = None if adaptive_M else int(args.M)
    M_label = f"auto/{args.selector}" if adaptive_M else str(M_fixed)
    logger.info("M mode: %s (pr_gamma=%s, pr_subtract_min=%s)",
                M_label, args.pr_gamma, args.pr_subtract_min)

    # ── Load model ──
    from transformers import AutoProcessor, AutoModelForImageTextToText
    logger.info("Loading %s ...", args.vlm_model)
    kw = dict(torch_dtype=torch.bfloat16)
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model, attn_implementation="sdpa", **kw)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(args.vlm_model, **kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    proc_kw = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    processor = AutoProcessor.from_pretrained(args.vlm_model, **proc_kw)
    letter_variants = get_letter_token_id_variants(processor.tokenizer, LETTERS)

    # ── Load samples ──
    if args.benchmark == "video_mme_v2":
        all_samples = load_video_mme_v2(args.data_dir, with_subtitle=args.with_subtitle)
        # Allow filtering by level (1/2/3) via --duration_bin (compat shim)
    elif args.benchmark == "lvb":
        all_samples = load_lvb(args.lvb_json, lvb_dir=args.lvb_dir,
                               with_subtitle=args.with_subtitle)
        logger.info("LVB: loaded %d samples from %s (with_subtitle=%s)",
                    len(all_samples), args.lvb_json, args.with_subtitle)
    else:
        all_samples = load_video_mme(args.data_dir)
    if args.duration_bin:
        all_samples = [s for s in all_samples if s["duration_bin"] == args.duration_bin]
    # ── Sharding for parallel runs (interleaved stride) ──
    if args.n_shards > 1:
        if not (0 <= args.shard_index < args.n_shards):
            raise ValueError(
                f"--shard_index={args.shard_index} must be in [0, {args.n_shards})")
        before = len(all_samples)
        all_samples = all_samples[args.shard_index::args.n_shards]
        logger.info("Sharding: shard %d/%d → %d / %d samples",
                    args.shard_index, args.n_shards, len(all_samples), before)
    # n_samples <= 0 means "all"
    if args.n_samples and args.n_samples > 0:
        samples = all_samples[:args.n_samples]
    else:
        samples = all_samples
    logger.info("Running %d samples (K=%d, M=%s)", len(samples), K, M_label)

    # ── Counters: aggregated overall + per bin ──
    # v1: bins are duration (short/medium/long); v2: bins are level ("1","2","3")
    METHODS = ["baseline_full", "baseline_uniform_M", "baseline_random_M",
               "probe_ensemble", "two_stage"]
    if args.benchmark == "video_mme_v2":
        BINS = BINS_V2 + ["0"]  # "0" catches non-level (first-3-of-4 grouped) questions
    elif args.benchmark == "lvb":
        BINS = BINS_LVB        # ["15","60","600","3600"] — duration_group buckets
    else:
        BINS = ["short", "medium", "long"]
    TIMING_KEYS = ["stage1", "stage2", "baseline", "uniform", "random"]

    def fresh_correct():
        return {m: 0 for m in METHODS}
    def fresh_timings():
        return {k: [] for k in TIMING_KEYS}
    def fresh_flops():
        return {m: [] for m in METHODS}

    correct        = fresh_correct()
    correct_by_bin = {b: fresh_correct() for b in BINS}
    total          = 0
    total_by_bin   = {b: 0 for b in BINS}

    timings        = fresh_timings()
    timings_by_bin = {b: fresh_timings() for b in BINS}

    flops          = fresh_flops()
    flops_by_bin   = {b: fresh_flops() for b in BINS}

    all_results = []
    adaptive_Ms = []  # tracks adaptive M per sample

    # Analytical-FLOPs config (single-model: same model used for probe and focused)
    from GridProbe.eval.two_stage_eval_crossmodel import (
        infer_lm_dims, lm_flops_per_forward,
    )
    model_dims = infer_lm_dims(model)
    logger.info("LM dims for FLOPs accounting: %s", model_dims)

    # ── Resume from existing --output if present ──
    # On crash-and-resume, we replay the existing results into the running
    # counters, then skip those sample IDs in the loop below.
    done_ids = set()
    if not args.no_resume and Path(args.output).exists():
        try:
            with open(args.output) as _f_resume:
                _existing = json.load(_f_resume)
            _existing_results = _existing.get("results") or []
            for r in _existing_results:
                rid = r.get("id")
                if rid is None:
                    continue
                bin_name = r.get("duration_bin", "?")
                # Replay counters for this sample
                if bin_name in BINS:
                    total += 1
                    total_by_bin[bin_name] += 1
                    gold = r.get("answer", "")
                    for m in METHODS:
                        pred = r.get(m)
                        if pred is not None and pred == gold:
                            correct[m] += 1
                            correct_by_bin[bin_name][m] += 1
                        f_val = r.get(f"flops_{m}")
                        if f_val is not None:
                            flops[m].append(f_val)
                            flops_by_bin[bin_name][m].append(f_val)
                    # Replay timings
                    for tk in TIMING_KEYS:
                        tv = r.get(f"t_{tk}")
                        if tv is not None:
                            timings[tk].append(tv)
                            timings_by_bin[bin_name][tk].append(tv)
                if "M_used" in r:
                    adaptive_Ms.append(r["M_used"])
                all_results.append(r)
                done_ids.add(rid)
            if done_ids:
                logger.info("Resumed from %s: %d samples already done",
                            args.output, len(done_ids))
        except Exception as e:
            logger.warning("Could not resume from %s: %s. Starting fresh.",
                           args.output, e)
            done_ids = set()

    # ── Helper: atomic save of the full eval state ──
    # Defined here (early) so checkpointing can call it from inside the loop.
    # Self-contained: uses a local _avg_t, doesn't depend on later helpers.
    _TIME_KEY_FOR = {
        "baseline_full":      "baseline",
        "baseline_uniform_M": "uniform",
        "baseline_random_M":  "random",
        "probe_ensemble":     "stage1",
        "two_stage":          None,  # = stage1 + stage2
    }
    def _avg_t(t_block, m):
        if m == "two_stage":
            s1 = float(np.mean(t_block["stage1"])) if t_block.get("stage1") else 0.0
            s2 = float(np.mean(t_block["stage2"])) if t_block.get("stage2") else 0.0
            return s1 + s2
        tk = _TIME_KEY_FOR[m]
        return float(np.mean(t_block[tk])) if t_block.get(tk) else 0.0

    def _save_state():
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w") as _f:
            json.dump({
                "benchmark": args.benchmark,
                "K": K, "n_frames": n_frames,
                "M_mode": M_label,
                "selector": args.selector,
                "pr_gamma": args.pr_gamma,
                "pr_subtract_min": args.pr_subtract_min,
                "max_video_pixels_probe": args.max_video_pixels_probe,
                "max_video_pixels_focus": args.max_video_pixels_focus,
                "model_dims": model_dims,
                "total":          total,
                "total_by_bin":   total_by_bin,
                "correct":        correct,
                "correct_by_bin": correct_by_bin,
                "avg_time_overall":  {m: _avg_t(timings, m) for m in METHODS},
                "avg_time_by_bin":   {b: {m: _avg_t(timings_by_bin[b], m)
                                           for m in METHODS} for b in BINS},
                "avg_tflops_overall":{m: (float(np.mean(flops[m])) / 1e12 if flops[m] else 0.0)
                                       for m in METHODS},
                "avg_tflops_by_bin": {b: {m: (float(np.mean(flops_by_bin[b][m])) / 1e12
                                                if flops_by_bin[b][m] else 0.0)
                                           for m in METHODS} for b in BINS},
                "adaptive_Ms": adaptive_Ms,
                "results": all_results,
            }, _f, indent=2)
        tmp.replace(out)

    for si, sample in enumerate(samples):
        if sample["id"] in done_ids:
            # Already processed in a prior run; counters were replayed at startup
            continue
        logger.info("[%d/%d] %s [%s]", si + 1, len(samples),
                    sample["id"][:20], sample["duration_bin"])

        frames = extract_frames_uniform(sample["video_path"], n_frames)
        if frames is None:
            logger.warning("  Frame extraction failed")
            continue

        if args.benchmark == "video_mme_v2":
            prompt_text = build_prompt(
                sample["question"], sample["options"],
                subtitle=sample.get("subtitle") if args.with_subtitle else None)
        elif args.benchmark == "lvb":
            prompt_text = build_prompt(
                sample["question"], sample["options"],
                subtitle=sample.get("subtitle") if args.with_subtitle else None)
        else:
            prompt_text = build_prompt(sample["question"], sample["options"])
        answer = sample["answer"]
        options = sample.get("options", [])

        try:
            # ════════════════════════════════════════════
            # Try shared cache (probe + baselines)
            # ════════════════════════════════════════════
            cached = load_shared_cache(
                args.probe_cache_dir, sample, K, args.max_video_pixels_probe)

            if cached is not None:
                # Reconstruct probe dict from cache
                row_lp = cached["row_lp"]
                col_lp = cached["col_lp"]
                row_confs = cached["row_confs"]
                col_confs = cached["col_confs"]
                M_mult = row_confs[:, None] * col_confs[None, :]
                flat = M_mult.flatten()
                sorted_idx = np.argsort(flat)[::-1]
                golden = [(int(idx // K), int(idx % K)) for idx in sorted_idx]
                probe = {
                    "row_lp": row_lp, "col_lp": col_lp,
                    "row_confs": row_confs, "col_confs": col_confs,
                    "row_preds": cached["row_preds"],
                    "col_preds": cached["col_preds"],
                    "M_mult": M_mult, "golden": golden,
                    "ensemble_pred": cached["ensemble_pred"],
                }
                base_full = {"pred": cached["base_full_pred"], "lp": cached["base_full_lp"]}
                base_uniform = {"pred": cached["base_uniform_pred"],
                                 "lp": cached["base_uniform_lp"]}
                base_random = {"pred": cached["base_random_pred"],
                                "lp": cached["base_random_lp"]}
                t_stage1 = cached.get("t_stage1", 0.0)
                t_base = cached.get("t_base", 0.0)
                t_uniform = cached.get("t_uniform", 0.0)
                t_random = cached.get("t_random", 0.0)
                timings["stage1"].append(t_stage1)
                timings["baseline"].append(t_base)
                timings["uniform"].append(t_uniform)
                timings["random"].append(t_random)
                logger.debug("  cache HIT")
            else:
                # ──── Stage 1: Grid probe (row + col passes) ────
                t0 = time.perf_counter()
                probe = stage1_probe(
                    model, processor, prompt_text, frames, K,
                    letter_variants, device, args.max_video_pixels_probe)
                t_stage1 = time.perf_counter() - t0
                timings["stage1"].append(t_stage1)

                # ──── Baselines ────
                if args.skip_full_baseline:
                    base_full = {"pred": "(skip)", "lp": None}
                    t_base = 0.0
                else:
                    t0 = time.perf_counter()
                    base_full = run_baseline_full(
                        model, processor, prompt_text, frames,
                        letter_variants, device, args.max_video_pixels_focus)
                    t_base = time.perf_counter() - t0
                    timings["baseline"].append(t_base)

                if args.skip_matched_baselines:
                    base_uniform = {"pred": "(skip)", "lp": None}
                    base_random  = {"pred": "(skip)", "lp": None}
                    t_uniform = 0.0
                    t_random  = 0.0
                else:
                    t0 = time.perf_counter()
                    base_uniform = run_baseline_uniform(
                        model, processor, prompt_text, frames, n_frames,
                        letter_variants, device, args.max_video_pixels_focus)
                    t_uniform = time.perf_counter() - t0
                    timings["uniform"].append(t_uniform)

                    t0 = time.perf_counter()
                    base_random = run_baseline_random(
                        model, processor, prompt_text, frames, n_frames,
                        letter_variants, device, args.max_video_pixels_focus)
                    t_random = time.perf_counter() - t0
                    timings["random"].append(t_random)

                # Save shared cache (only if dir specified)
                save_shared_cache(
                    args.probe_cache_dir, sample, K, args.max_video_pixels_probe,
                    {
                        "row_lp": probe["row_lp"], "col_lp": probe["col_lp"],
                        "row_confs": probe["row_confs"], "col_confs": probe["col_confs"],
                        "row_preds": probe["row_preds"], "col_preds": probe["col_preds"],
                        "ensemble_pred": probe["ensemble_pred"],
                        "base_full_pred": base_full["pred"], "base_full_lp": base_full["lp"],
                        "base_uniform_pred": base_uniform["pred"],
                        "base_uniform_lp": base_uniform.get("lp", None),
                        "base_random_pred": base_random["pred"],
                        "base_random_lp": base_random.get("lp", None),
                        "t_stage1": t_stage1, "t_base": t_base,
                        "t_uniform": t_uniform, "t_random": t_random,
                    })

            # ════════════════════════════════════════════
            # Resolve M (depends on selector — always fresh)
            # ════════════════════════════════════════════
            if adaptive_M:
                M = select_M(args.selector, probe["M_mult"], args)
            else:
                M = M_fixed
            adaptive_Ms.append(M)

            # ════════════════════════════════════════════
            # Stage 2: Focused pass on golden cells (always fresh)
            # ════════════════════════════════════════════
            t0 = time.perf_counter()
            focused = stage2_focused(
                model, processor, prompt_text, frames,
                probe["golden"], M, K,
                letter_variants, device, args.max_video_pixels_focus)
            t_stage2 = time.perf_counter() - t0
            timings["stage2"].append(t_stage2)

            # ════════════════════════════════════════════
            # Check answers
            # ════════════════════════════════════════════
            c_base = check_answer(base_full["pred"], answer, options)
            c_uniform = check_answer(base_uniform["pred"], answer, options)
            c_random = check_answer(base_random["pred"], answer, options)
            c_probe = check_answer(probe["ensemble_pred"], answer, options)
            c_twostage = check_answer(focused["pred"], answer, options)

            # ════════════════════════════════════════════
            # Token counts + analytical FLOPs
            # We peek build_inputs to get LM seq length per pass.
            # ════════════════════════════════════════════
            try:
                _peek_full   = build_inputs(processor, prompt_text, frames,
                                            args.max_video_pixels_focus)
                n_tok_full   = int(_peek_full["input_ids"].shape[1])
            except Exception:
                n_tok_full   = -1
            try:
                _peek_uni    = build_inputs(processor, prompt_text,
                                            [frames[i] for i in
                                             np.linspace(0, len(frames)-1, n_frames)
                                                  .round().astype(int).tolist()],
                                            args.max_video_pixels_focus)
                n_tok_uni    = int(_peek_uni["input_ids"].shape[1])
            except Exception:
                n_tok_uni    = n_tok_full
            n_tok_random     = n_tok_uni  # same frame count as uniform-K²
            try:
                golden_frames = [frames[fi] for fi in focused["frame_indices"]]
                _peek_2s     = build_inputs(processor, prompt_text, golden_frames,
                                            args.max_video_pixels_focus)
                n_tok_2s     = int(_peek_2s["input_ids"].shape[1])
            except Exception:
                n_tok_2s     = -1
            try:
                _peek_probe  = build_inputs(processor, prompt_text,
                                            [frames[i] for i in row_indices(K)[0]],
                                            args.max_video_pixels_probe)
                n_tok_probe_pp = int(_peek_probe["input_ids"].shape[1])
            except Exception:
                n_tok_probe_pp = -1

            f_full   = lm_flops_per_forward(n_tok_full,   model_dims)
            f_uni    = lm_flops_per_forward(n_tok_uni,    model_dims)
            f_random = lm_flops_per_forward(n_tok_random, model_dims)
            f_probe  = 2 * K * lm_flops_per_forward(n_tok_probe_pp, model_dims)
            f_2s     = f_probe + lm_flops_per_forward(n_tok_2s, model_dims)

            # ════════════════════════════════════════════
            # Tally — overall + per bin
            # ════════════════════════════════════════════
            bin_name = sample.get("duration_bin", "?")
            if bin_name not in BINS:
                # Unknown bin (rare): keep tally only in overall.
                bin_name = None

            def _tally(method, c, t, fl):
                correct[method] += c
                timings_key = {"baseline_full": "baseline",
                               "baseline_uniform_M": "uniform",
                               "baseline_random_M": "random",
                               "probe_ensemble": "stage1",
                               "two_stage": "stage2"}[method]
                # 'two_stage' time is t_stage2 (focused only); for breakdown we
                # use t_stage1 + t_stage2 in the summary block. Stash into stage2.
                # (The probe time was already appended above; do not double-count.)
                # We only use this helper for FLOPs and per-bin counts.
                flops[method].append(fl)
                if bin_name is not None:
                    correct_by_bin[bin_name][method] += c
                    flops_by_bin[bin_name][method].append(fl)

            _tally("baseline_full",      c_base,     t_base,    f_full)
            _tally("baseline_uniform_M", c_uniform,  t_uniform, f_uni)
            _tally("baseline_random_M",  c_random,   t_random,  f_random)
            _tally("probe_ensemble",     c_probe,    t_stage1,  f_probe)
            _tally("two_stage",          c_twostage, t_stage2,  f_2s)
            total += 1
            if bin_name is not None:
                total_by_bin[bin_name] += 1
                # Per-bin timings
                timings_by_bin[bin_name]["stage1"].append(t_stage1)
                timings_by_bin[bin_name]["stage2"].append(t_stage2)
                timings_by_bin[bin_name]["baseline"].append(t_base)
                timings_by_bin[bin_name]["uniform"].append(t_uniform)
                timings_by_bin[bin_name]["random"].append(t_random)

            logger.info(
                "  [%s] ans=%s M=%d | base=%s(%s,%.1fT) uni=%s(%s,%.1fT) rnd=%s(%s,%.1fT) "
                "probe=%s(%s,%.1fT) 2stage=%s(%s,%.1fT) | t1=%.1fs t2=%.1fs base=%.1fs",
                bin_name or "?", answer, M,
                base_full["pred"],    "OK" if c_base else "XX",     f_full/1e12,
                base_uniform["pred"], "OK" if c_uniform else "XX",  f_uni/1e12,
                base_random["pred"],  "OK" if c_random else "XX",   f_random/1e12,
                probe["ensemble_pred"], "OK" if c_probe else "XX",  f_probe/1e12,
                focused["pred"],      "OK" if c_twostage else "XX", f_2s/1e12,
                t_stage1, t_stage2, t_base)

            all_results.append({
                "id": sample["id"],
                "duration_bin": sample["duration_bin"],
                "question": sample["question"][:80],
                "answer": answer,
                "M_used": M,
                "baseline_full": base_full["pred"],
                "baseline_uniform_M": base_uniform["pred"],
                "baseline_random_M": base_random["pred"],
                "probe_ensemble": probe["ensemble_pred"],
                "two_stage": focused["pred"],
                "golden_frames": focused["frame_indices"],
                "row_confs": probe["row_confs"].tolist(),
                "col_confs": probe["col_confs"].tolist(),
                "t_stage1": t_stage1, "t_stage2": t_stage2,
                "t_baseline": t_base, "t_uniform": t_uniform, "t_random": t_random,
                "n_tok_baseline_full": n_tok_full,
                "n_tok_uniform_M":     n_tok_uni,
                "n_tok_random_M":      n_tok_random,
                "n_tok_two_stage":     n_tok_2s,
                "n_tok_probe_per_pass": n_tok_probe_pp,
                "flops_baseline_full":      f_full,
                "flops_baseline_uniform_M": f_uni,
                "flops_baseline_random_M":  f_random,
                "flops_probe_ensemble":     f_probe,
                "flops_two_stage":          f_2s,
            })

        except Exception as e:
            logger.warning("  Failed: %s", e)
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
        finally:
            # ── Per-sample memory cleanup to prevent CPU-RAM creep ──
            # PIL frames, PyTorch graph buffers, and decord state can accumulate
            # across iterations and trigger the OOM-killer in long parallel runs.
            try:
                del frames
            except Exception:
                pass
            try:
                del probe
            except Exception:
                pass
            try:
                del focused
            except Exception:
                pass
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Periodic checkpoint: write partial JSON every N samples so a crash
        # mid-loop is recoverable. Uses atomic temp+rename via _save_state().
        if args.checkpoint_every > 0 and (si + 1) % args.checkpoint_every == 0:
            try:
                _save_state()
                logger.info("  [checkpoint] saved %d/%d to %s",
                            si + 1, len(samples), args.output)
            except Exception as _e:
                logger.warning("  [checkpoint failed] %s", _e)

    # ════════════════════════════════════════════
    # Summary tables: per-bin and overall, with TFLOPs
    # ════════════════════════════════════════════
    M_display = f"auto (PR, avg={np.mean(adaptive_Ms):.1f})" if adaptive_M and adaptive_Ms else str(M_fixed)
    M_label_short = f"auto≈{np.mean(adaptive_Ms):.0f}" if adaptive_M and adaptive_Ms else str(M_fixed)

    method_labels = {
        "baseline_full":      "baseline (K² frames)",
        "baseline_uniform_M": f"uniform-K² (={n_frames})",
        "baseline_random_M":  f"random-K² (={n_frames})",
        "probe_ensemble":     "probe ensemble",
        "two_stage":          f"TWO-STAGE (M={M_label_short})",
    }
    time_key_for = {
        "baseline_full":      "baseline",
        "baseline_uniform_M": "uniform",
        "baseline_random_M":  "random",
        "probe_ensemble":     "stage1",
        "two_stage":          None,  # special: stage1 + stage2
    }

    def _avg_time(t_block, m):
        if m == "two_stage":
            avg_s1 = float(np.mean(t_block["stage1"])) if t_block.get("stage1") else 0.0
            avg_s2 = float(np.mean(t_block["stage2"])) if t_block.get("stage2") else 0.0
            return avg_s1 + avg_s2
        tk = time_key_for[m]
        return float(np.mean(t_block[tk])) if t_block.get(tk) else 0.0

    def fmt_block(n, c_block, t_block, f_block, header):
        if n <= 0:
            print(f"  ({header}: no samples)")
            return
        print(f"  ── {header}  (n={n}) ──")
        print(f"  {'Method':<35} {'Acc':>7} {'Correct':>10} {'Avg time':>10} {'Avg TFLOPs':>11}")
        print("  " + "-" * 80)
        for m in METHODS:
            acc = c_block[m] / n * 100 if n else 0.0
            avg_t = _avg_time(t_block, m)
            f_list = f_block.get(m, [])
            avg_f = float(np.mean(f_list)) if f_list else 0.0
            t_str = f"{avg_t:.2f}s"
            if m == "two_stage":
                t_str += " (s1+s2)"
            print(f"  {method_labels[m]:<35} {acc:>6.1f}%  {c_block[m]:>4}/{n:<4} {t_str:>9} {avg_f/1e12:>10.2f}T")
        print()

    print("\n" + "=" * 92)
    print(f"  TWO-STAGE GOLDEN-CELL EVAL (benchmark={args.benchmark}, K={K}, M={M_display}, n={total})")
    print("=" * 92)

    # Per-bin blocks
    for bn in BINS:
        # For v2, label level "0" as 'unleveled' for clarity
        header_label = bn
        if args.benchmark == "video_mme_v2":
            header_label = (f"LEVEL {bn}" if bn in ("1", "2", "3") else "UNLEVELED (group q1-q3)")
        else:
            header_label = bn.upper()
        fmt_block(total_by_bin[bn], correct_by_bin[bn],
                  timings_by_bin[bn], flops_by_bin[bn],
                  header_label)

    # Overall block
    fmt_block(total, correct, timings, flops, "OVERALL")

    # Headline deltas
    if total > 0:
        ts_acc = correct["two_stage"] / total * 100
        base_acc = correct["baseline_full"] / total * 100
        uni_acc = correct["baseline_uniform_M"] / total * 100
        print(f"  Δ TWO-STAGE vs baseline (K²={n_frames}):  "
              f"{ts_acc - base_acc:+.1f}% (overall)")
        for bn in BINS:
            if total_by_bin[bn] == 0:
                continue
            ts_b = correct_by_bin[bn]["two_stage"]    / total_by_bin[bn] * 100
            b_b  = correct_by_bin[bn]["baseline_full"] / total_by_bin[bn] * 100
            print(f"      {bn:>10}: {ts_b - b_b:+.1f}%")
        print(f"  Δ TWO-STAGE vs uniform-K²:  {ts_acc - uni_acc:+.1f}% (overall)")

        # FLOPs ratios
        if flops["two_stage"] and flops["baseline_full"]:
            avg_ts   = float(np.mean(flops["two_stage"]))
            avg_base = float(np.mean(flops["baseline_full"]))
            print(f"  TFLOPs ratio (two_stage / baseline_full):  {avg_ts/avg_base:.2f}×")
        # Time ratio (two_stage = stage1+stage2 vs baseline)
        avg_ts_t = (np.mean(timings["stage1"]) + np.mean(timings["stage2"])) \
                    if timings["stage1"] and timings["stage2"] else 0.0
        avg_b_t  = float(np.mean(timings["baseline"])) if timings["baseline"] else 0.0
        if avg_ts_t > 0 and avg_b_t > 0:
            print(f"  Wall-clock ratio (two_stage / baseline_full): {avg_ts_t/avg_b_t:.2f}×")

        # Adaptive M distribution
        if adaptive_M and adaptive_Ms:
            Ms = np.array(adaptive_Ms)
            print()
            print(f"  Adaptive M stats: min={Ms.min()}  median={int(np.median(Ms))}  "
                  f"mean={Ms.mean():.1f}  max={Ms.max()}  std={Ms.std():.1f}")
            print(f"  M distribution:")
            for m_val in sorted(set(Ms.tolist())):
                count = int((Ms == m_val).sum())
                bar = "█" * min(count, 60)
                print(f"    M={m_val:>3}: {bar} ({count})")
    print("=" * 92)

    # Save (uses the same atomic _save_state helper as the periodic checkpoint)
    _save_state()
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
