"""
Grid Sampled Ensemble — Training-Free Evaluation
==================================================
Instead of masking attention on a K²-frame sequence, run MANY small forward
passes, each with only K frames drawn along one axis of the conceptual K×K
grid — then ensemble the answer-letter log-probs.

For K = 8 (so grid of 8×8 = 64 frames):
  • 8 ROW passes:   frames [r*K : r*K + K]  for r in 0..K-1  (local temporal)
  • 8 COL passes:   frames [c, c+K, c+2K, ...] for c in 0..K-1  (periodicity)
  • 8 DIAG passes:  frames along each grid diagonal (steady progression)

Each pass only asks the VLM to process K frames, so attention cost is
O((K·tpf)²) instead of O((K²·tpf)²) — ~1/K² per pass, ~3/K total, cheaper
than one full-frame baseline at K≥4.

For MCQ benchmarks (Video-MME, LongVideoBench) we score the letter tokens
after the assistant prefix and sum log-probs across passes → argmax.
No generate() loop, no attention masking, no size-mismatch headaches.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m GridProbe.eval.grid_sampled_ensemble_eval \
        --benchmark video_mme --data_dir /path/to/Video-MME \
        --n_frames 64 --n_samples 100 --axes row,col,diag --debug

    torchrun --nproc_per_node 4 -m GridProbe.eval.grid_sampled_ensemble_eval \
        --benchmark lvb --data_dir /path/to/LongVideoBench \
        --n_frames 64 --n_samples 400 --axes row,col,diag
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.video_mme import VideoMMEEvaluator, build_prompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Axis index selection on the K×K grid
# ═══════════════════════════════════════════════════════════════

def row_indices(K: int):
    """K rows, each of length K. Returns list of K index lists."""
    return [list(range(r * K, r * K + K)) for r in range(K)]


def col_indices(K: int):
    """K columns, each of length K."""
    return [[c + r * K for r in range(K)] for c in range(K)]


def diag_indices(K: int):
    """K diagonals (wrap-around), each of length K.
    Diagonal d: frames (r, (r+d) mod K) for r in 0..K-1."""
    return [[r * K + ((r + d) % K) for r in range(K)] for d in range(K)]


def antidiag_indices(K: int):
    return [[r * K + ((d - r) % K) for r in range(K)] for d in range(K)]


def block_indices(K: int):
    """K-1 overlapping 2×2 blocks along the main diagonal.

    Each block picks 4 frames:
      (r, c), (r, c+1), (r+1, c), (r+1, c+1)
    → two consecutive frames (local continuity, like row)
    → from two consecutive grid rows (stride-K echo, like col)
    → CROSS-SCALE signal: short-range + long-range in one pass.

    This is the natural third axis — analogous to a 2×2 CNN kernel on the
    K×K temporal grid. Each pass sees 4 frames arranged as
    {t, t+1, t+K, t+K+1} in original video time.

    Returns K-1 index lists of 4 frames each (one fewer pass than row/col).
    """
    if K < 2:
        return [list(range(K * K))]
    blocks = []
    for d in range(K - 1):
        r, c = d, d
        block = [r * K + c, r * K + c + 1, (r + 1) * K + c, (r + 1) * K + c + 1]
        blocks.append(block)
    return blocks


def strip_indices(K: int):
    """K-1 overlapping 2-row strips (2K frames each).

    Each strip = two consecutive grid rows = 2K consecutive frames.
    This is the "each row uses prev-row info" variant — stacked rows.
    More expensive than block (2K frames/pass vs 4), but richer per-pass context.
    """
    if K < 2:
        return [list(range(K * K))]
    return [list(range(r * K, (r + 2) * K)) for r in range(K - 1)]


AXIS_FNS = {
    "row": row_indices,
    "col": col_indices,
    "diag": diag_indices,
    "antidiag": antidiag_indices,
    "block": block_indices,
    "strip": strip_indices,
}


# ═══════════════════════════════════════════════════════════════
# Merge strategies — ALL training-free, evaluated in parallel
# ═══════════════════════════════════════════════════════════════

def merge_mean(lp_matrix):
    """Uniform mean of log-probs. Simple, noise-prone."""
    return lp_matrix.mean(dim=0)


def merge_sum(lp_matrix):
    """Sum of log-probs (product-of-experts). Same argmax as mean."""
    return lp_matrix.sum(dim=0)


def merge_conf_weighted(lp_matrix, tau: float = 1.0):
    """Weight each pass by its confidence (max log-prob). Confident passes dominate.

    w_i ∝ exp(max_j lp_ij / tau)  — softmax over max-log-prob across passes.
    Final = Σ_i w_i * lp_i
    """
    max_lp = lp_matrix.max(dim=1).values  # (n_passes,)
    weights = F.softmax(max_lp / tau, dim=0)
    return (weights.unsqueeze(1) * lp_matrix).sum(dim=0)


def merge_entropy_weighted(lp_matrix, tau: float = 1.0):
    """Weight by negative entropy. Low-entropy (peaky) passes get more weight."""
    probs = lp_matrix.exp()
    H = -(probs * lp_matrix).sum(dim=1)  # entropy per pass (nats)
    weights = F.softmax(-H / tau, dim=0)
    return (weights.unsqueeze(1) * lp_matrix).sum(dim=0)


def merge_topk(lp_matrix, k: int = None):
    """Use only the top-k most confident passes (by max log-prob), mean their log-probs."""
    if k is None:
        k = max(1, lp_matrix.shape[0] // 3)  # default: top third
    max_lp = lp_matrix.max(dim=1).values
    top_idx = max_lp.topk(min(k, lp_matrix.shape[0])).indices
    return lp_matrix[top_idx].mean(dim=0)


def merge_vote(lp_matrix):
    """Hard vote: each pass votes its argmax letter; return a pseudo-log-prob."""
    votes = lp_matrix.argmax(dim=1)  # (n_passes,)
    n_letters = lp_matrix.shape[1]
    counts = torch.bincount(votes, minlength=n_letters).float()
    # Add the confidence signal to break ties (log of count + mean log-prob per letter)
    return torch.log(counts + 1e-6) + 0.1 * lp_matrix.mean(dim=0)


def merge_centered_mean(lp_matrix):
    """Subtract each pass's own mean log-prob before averaging.
    Removes systematic per-letter bias (e.g. short-context 'A' prior)."""
    centered = lp_matrix - lp_matrix.mean(dim=1, keepdim=True)
    return centered.mean(dim=0)


def merge_centered_conf(lp_matrix, tau: float = 1.0):
    """Confidence-weighted mean after per-pass centering."""
    centered = lp_matrix - lp_matrix.mean(dim=1, keepdim=True)
    max_lp = centered.max(dim=1).values
    weights = F.softmax(max_lp / tau, dim=0)
    return (weights.unsqueeze(1) * centered).sum(dim=0)


MERGE_FNS = {
    "mean": merge_mean,
    "conf": merge_conf_weighted,
    "ent": merge_entropy_weighted,
    "topk": merge_topk,
    "vote": merge_vote,
    "cmean": merge_centered_mean,
    "cconf": merge_centered_conf,
}


# ═══════════════════════════════════════════════════════════════
# Frame extraction (once per video, reused across all passes)
# ═══════════════════════════════════════════════════════════════

def extract_frames_uniform(video_path: str, n_frames: int):
    """Extract n_frames uniformly-spaced PIL frames from the video.

    Uses decord if available (fast), else cv2 fallback.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total == 0:
            return None
        idx = np.linspace(0, total - 1, n_frames).round().astype(int).tolist()
        batch = vr.get_batch(idx).asnumpy()  # (n_frames, H, W, 3)
        from PIL import Image
        return [Image.fromarray(f) for f in batch]
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        idx = np.linspace(0, total - 1, n_frames).round().astype(int).tolist()
        frames = []
        from PIL import Image
        target = set(idx)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in target:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            i += 1
        cap.release()
        if len(frames) < n_frames and frames:
            # Pad by repeating last frame
            frames += [frames[-1]] * (n_frames - len(frames))
        return frames
    except Exception as e:
        logger.debug("Frame extract failed for %s: %s", video_path, e)
        return None


# ═══════════════════════════════════════════════════════════════
# Shared inputs builder (processor + chat template)
# ═══════════════════════════════════════════════════════════════

def build_inputs(processor, prompt_text, frames, max_video_pixels):
    """Run processor + chat template once. Returns the `inputs` dict (on CPU)."""
    from qwen_vl_utils import process_vision_info

    video_item = {"type": "video", "video": frames}
    if max_video_pixels and max_video_pixels > 0:
        side = int(math.sqrt(max_video_pixels))
        side = (side // 28) * 28
        side = max(56, side)
        video_item["resized_height"] = side
        video_item["resized_width"] = side

    messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ip = getattr(processor, "image_processor", None)
    ps = getattr(ip, "patch_size", None)
    pk = {"return_video_kwargs": True}
    if ps is not None:
        pk["image_patch_size"] = int(ps)

    out = process_vision_info(messages, **pk)
    if len(out) == 4:
        ii, vi, vk, vm = out
    else:
        ii, vi, vk = out
        vm = None

    ppk = dict(vk or {})
    if vk:
        ppk["do_resize"] = False
    if vm is not None:
        ppk["video_metadata"] = vm

    # Unwrap single-element list kwargs that newer Qwen3-VL processor expects as scalar.
    # qwen_vl_utils returns e.g. fps=[2.0] (list-per-video); new transformers wants fps=2.0.
    # We only unwrap scalar-ish keys (not video_metadata, which IS a list of per-video dicts).
    SCALAR_KEYS = {"fps", "max_pixels", "min_pixels", "patch_size", "temporal_patch_size",
                   "merge_size", "image_patch_size"}
    for k in list(ppk.keys()):
        v = ppk[k]
        if k in SCALAR_KEYS and isinstance(v, list) and len(v) == 1:
            ppk[k] = v[0]

    return processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt", **ppk)


def letters_log_probs_from_logits(logits, letter_variants, device):
    """Convert next-token logits → per-letter log-probs (logsumexp over variants)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    letter_lp = torch.empty(len(letter_variants), device=device, dtype=torch.float32)
    for i, ids in enumerate(letter_variants):
        if not ids:
            letter_lp[i] = float("-inf")
        else:
            idx = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
            letter_lp[i] = torch.logsumexp(log_probs[idx], dim=0)
    return letter_lp


# ═══════════════════════════════════════════════════════════════
# ViT visual-feature cache (run once per video, reused by all grid passes)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_visual_cache(model, processor, all_frames, max_video_pixels, device):
    """Run processor + ViT ONCE for all K² frames. Returns:
      cache_3d: (T_cache, tokens_per_group, D_v) — per temporal-group visual features
      tpt: temporal_patch_size (frames per group, typically 2 for Qwen3-VL)
    The grid passes will slice this instead of re-running the ViT.
    """
    # We only need visual features here; prompt text doesn't matter
    inputs = build_inputs(processor, "Describe.", all_frames, max_video_pixels)
    pvv = inputs.get("pixel_values_videos")
    vgt = inputs.get("video_grid_thw")
    if pvv is None or vgt is None:
        return None, None
    pvv = pvv.to(device)
    vgt = vgt.to(device)

    # Run ONLY the vision tower
    visual = getattr(model, "visual", None)
    if visual is None:
        # Try nested
        m = getattr(model, "model", model)
        visual = getattr(m, "visual", None)
    if visual is None:
        raise RuntimeError("Could not locate model.visual")

    visual_out = visual(pvv, grid_thw=vgt)
    if isinstance(visual_out, tuple):
        visual_out = visual_out[0]

    T_cache = int(vgt[0, 0].item())  # number of temporal groups AFTER temporal merging
    N_total = visual_out.shape[0] if visual_out.dim() == 2 else visual_out.numel() // visual_out.shape[-1]
    tokens_per_group = N_total // max(T_cache, 1)
    D_v = visual_out.shape[-1]
    cache_3d = visual_out.reshape(T_cache, tokens_per_group, D_v)

    tpt = max(1, len(all_frames) // T_cache)  # temporal_patch_size
    return cache_3d, tpt


@torch.no_grad()
def score_letters_cached(model, processor, prompt_text, subset_frames, subset_indices,
                          cache_3d, tpt, letter_variants, device, max_video_pixels):
    """Score letters for a K-frame pass WITHOUT re-running the ViT.

    Workflow:
      1. Run processor on K frames → get input_ids + vgt_pass (placeholder count).
      2. DON'T pass pixel_values_videos — skip the ViT entirely.
      3. Pick T_pass temporal groups from the cache_3d using subset_indices.
      4. Splice those features into the input embeddings at the visual placeholder
         positions, then call the inner LM directly.
    """
    inputs = build_inputs(processor, prompt_text, subset_frames, max_video_pixels)
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    vgt_pass = inputs["video_grid_thw"].to(device)
    T_pass = int(vgt_pass[0, 0].item())

    # Pick cache temporal-group indices for this pass.
    # Pass-group g represents pass-frames [g*tpt : (g+1)*tpt] within the K subset.
    # In the ORIGINAL K²-frame indexing those are subset_indices[g*tpt],
    # which live in cache group (subset_indices[g*tpt] // tpt).
    # For row subsets this is exact; for col/diag it's an approximation (cache
    # group at frame i represents the pair (2i, 2i+1), not our non-contiguous pick).
    T_cache = cache_3d.shape[0]
    cache_idx = []
    K_subset = len(subset_indices)
    for g in range(T_pass):
        frame_pos = min(g * tpt, K_subset - 1)
        ci = min(subset_indices[frame_pos] // tpt, T_cache - 1)
        cache_idx.append(ci)

    cache_idx_t = torch.tensor(cache_idx, device=cache_3d.device, dtype=torch.long)
    selected = cache_3d.index_select(0, cache_idx_t)  # (T_pass, tpf_cache, D_v)
    selected_flat = selected.reshape(-1, selected.shape[-1])

    # Build input embeddings, splice visual features into placeholder positions
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)

    vid_id = getattr(model.config, "video_token_id", None)
    img_id = getattr(model.config, "image_token_id", None)
    mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    if vid_id is not None:
        mask |= (input_ids[0] == vid_id)
    if img_id is not None:
        mask |= (input_ids[0] == img_id)
    positions = mask.nonzero(as_tuple=True)[0]
    n_place = len(positions)
    n_feat = selected_flat.shape[0]
    n = min(n_place, n_feat)
    if n > 0:
        inputs_embeds[0, positions[:n]] = selected_flat[:n].to(inputs_embeds.dtype)

    # Call the inner LM directly — bypasses ViT and the visual-merger step
    inner = getattr(model, "model", model)
    lm_head = getattr(model, "lm_head", None)
    lm_out = inner(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
    hidden = lm_out[0] if isinstance(lm_out, tuple) else lm_out.last_hidden_state
    last_hidden = hidden[0, -1, :]
    if lm_head is not None:
        logits = lm_head(last_hidden)
    else:
        logits = last_hidden

    return letters_log_probs_from_logits(logits, letter_variants, device)


# ═══════════════════════════════════════════════════════════════
# Letter-scoring via one prefill per pass
# ═══════════════════════════════════════════════════════════════

def get_letter_token_id_variants(tokenizer, letters):
    """For each letter, collect ALL first-token ids of common surface forms.

    Qwen BPE: "A" and " A" and "(A" are different tokens. The first token after
    the assistant prefix can be any of these depending on prompt context, so we
    score all variants and take the max log-prob per letter.

    Returns: list[list[int]] — per-letter list of candidate token ids.
    """
    variants_per_letter = []
    for L in letters:
        ids = set()
        for surface in (L, f" {L}", f"({L}", f"{L}.", f" {L}.", f"{L})"):
            toks = tokenizer(surface, add_special_tokens=False).input_ids
            if len(toks) >= 1:
                ids.add(int(toks[0]))
        variants_per_letter.append(sorted(ids))
    return variants_per_letter


@torch.no_grad()
def score_letters_for_pass(model, processor, prompt_text, frames_subset, letter_variants, device,
                            max_video_pixels=0, return_topk=False):
    """Baseline path: run the FULL model (ViT + LM) on frames_subset.

    Used for the baseline K²-frame pass. Grid passes use `score_letters_cached`
    which skips the ViT by reusing a pre-computed visual cache.
    """
    inputs = build_inputs(processor, prompt_text, frames_subset, max_video_pixels)

    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pvv = inputs.get("pixel_values_videos")
    vgt = inputs.get("video_grid_thw")
    mm_tt = inputs.get("mm_token_type_ids")  # newer Qwen3-VL processor returns this
    if pvv is not None:
        pvv = pvv.to(device)
    if vgt is not None:
        vgt = vgt.to(device)
    if mm_tt is not None:
        mm_tt = mm_tt.to(device)

    kwargs = {"input_ids": input_ids, "attention_mask": attn, "use_cache": False}
    if pvv is not None:
        kwargs["pixel_values_videos"] = pvv
    if vgt is not None:
        kwargs["video_grid_thw"] = vgt
    if mm_tt is not None:
        kwargs["mm_token_type_ids"] = mm_tt

    outputs = model(**kwargs)
    logits = outputs.logits[0, -1, :]
    letter_lp = letters_log_probs_from_logits(logits, letter_variants, device)

    if return_topk:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        top = torch.topk(log_probs, k=5)
        topk_repr = [(int(tid.item()), float(lp.item())) for tid, lp in zip(top.indices, top.values)]
        return letter_lp, topk_repr
    return letter_lp


# ═══════════════════════════════════════════════════════════════
# Data loaders (shared with grid_attention_mask_eval)
# ═══════════════════════════════════════════════════════════════

def load_video_mme(data_dir):
    evaluator = VideoMMEEvaluator(data_dir)
    samples = []
    for s in evaluator.samples:
        video_path = evaluator._find_video(s["video_id"])
        if not video_path:
            continue
        samples.append({
            "video_path": video_path,
            "question": s["question"],
            "options": s.get("options", []),
            "answer": s["answer"],
            "duration_bin": s.get("duration_bin", "?"),
            "id": s.get("question_id", ""),
        })
    return samples


def load_lvb(data_dir):
    with open(os.path.join(data_dir, "lvb_val.json")) as f:
        data = json.load(f)
    samples = []
    for s in data:
        vp = os.path.join(data_dir, "videos", f"{s['video_id']}.mp4")
        if not os.path.exists(vp):
            continue
        options = s.get("candidates", [])
        correct = s.get("correct_choice", 0)
        answer = chr(65 + correct)
        samples.append({
            "video_path": vp,
            "question": s["question"],
            "options": options,
            "answer": answer,
            "duration_bin": str(s.get("duration_group", "?")),
            "id": s.get("id", ""),
        })
    return samples


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["video_mme", "lvb"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--n_frames", type=int, default=64,
                        help="Total frames per video (will be arranged as K×K, K=sqrt(n_frames))")
    parser.add_argument("--max_video_pixels", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--axes", default="row,col,diag",
                        help="Comma-separated: row,col,diag,antidiag")
    parser.add_argument("--duration_bin", default=None)
    parser.add_argument("--output", default="grid_sampled_eval.json")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--grid_only", action="store_true",
                        help="Skip the K²-frame baseline pass")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--no_vit_cache", action="store_true",
                        help="Disable ViT caching (re-encode frames every pass)")
    args = parser.parse_args()

    # ── Distributed setup ──
    is_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world_size = 0, 1

    logging.basicConfig(
        level=logging.DEBUG if (args.debug and rank == 0) else logging.INFO,
        format=f"%(asctime)s [rank{rank}] %(message)s",
    )

    # Grid dims
    K = int(round(math.sqrt(args.n_frames)))
    if K * K != args.n_frames:
        if rank == 0:
            logger.warning("n_frames=%d is not a perfect square; using K=%d → %d frames",
                           args.n_frames, K, K * K)
    n_total = K * K

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in AXIS_FNS:
            raise ValueError(f"Unknown axis: {a}")

    # ── Load VLM (SDPA default, no masking needed) ──
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if rank == 0:
        logger.info("Loading %s ...", args.vlm_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(args.vlm_model, cache_dir=args.cache_dir)
    model.eval()

    device = f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if rank == 0:
        logger.info("K=%d  grid=%dx%d  axes=%s", K, K, K, axes)
        logger.info("Per sample: 1 baseline (%d frames) + %d grid passes (%d frames each)",
                    n_total, len(axes) * K, K)

    # ── Load samples ──
    samples = load_video_mme(args.data_dir) if args.benchmark == "video_mme" else load_lvb(args.data_dir)
    if rank == 0:
        logger.info("Loaded %d samples", len(samples))
    if args.duration_bin:
        samples = [s for s in samples if s["duration_bin"] == args.duration_bin]
    if args.n_samples > 0:
        samples = samples[:args.n_samples]
    samples_per_rank = samples[rank::world_size]

    # ── Letter token variants ──
    letters_4 = ["A", "B", "C", "D"]
    letters_5 = ["A", "B", "C", "D", "E"]
    variants_4 = get_letter_token_id_variants(processor.tokenizer, letters_4)
    variants_5 = get_letter_token_id_variants(processor.tokenizer, letters_5)
    if rank == 0:
        tok = processor.tokenizer
        for L, ids in zip(letters_4, variants_4):
            logger.info("  letter '%s' → variants: %s",
                        L, [(i, repr(tok.decode([i]))) for i in ids])

    run_baseline = not args.grid_only
    run_grid = not args.baseline_only

    # Track baseline + one counter per merge strategy + per-axis + oracle
    local = {"baseline": 0, "total": 0, "by_dg": {}}
    for m in MERGE_FNS:
        local[f"grid_{m}"] = 0
    for a in axes:
        local[f"axis_{a}"] = 0
    local["oracle_axes"] = 0       # best of (row/col/diag) per sample
    local["oracle_all"] = 0        # best of (baseline, axes, best-merge) per sample

    iterator = tqdm(samples_per_rank, desc=f"Rank{rank}", disable=not args.debug)
    for sample in iterator:
        try:
            n_opt = len(sample["options"])
            if n_opt == 5:
                letters = letters_5
                variants = variants_5
            else:
                letters = letters_4
                variants = variants_4

            correct_letter = sample["answer"].upper()
            if correct_letter not in letters:
                continue
            correct_idx = letters.index(correct_letter)

            prompt_text = build_prompt(sample["question"], sample["options"])

            # Extract all K² frames ONCE
            all_frames = extract_frames_uniform(sample["video_path"], n_total)
            if all_frames is None or len(all_frames) < n_total:
                continue

            # ─── Baseline: all K² frames in one pass ───
            c_b = False
            pred_b_letter = "?"
            baseline_topk = None
            if run_baseline:
                if args.debug:
                    lp_b, baseline_topk = score_letters_for_pass(
                        model, processor, prompt_text, all_frames, variants, device,
                        max_video_pixels=args.max_video_pixels, return_topk=True)
                else:
                    lp_b = score_letters_for_pass(
                        model, processor, prompt_text, all_frames, variants, device,
                        max_video_pixels=args.max_video_pixels)
                pred_b_idx = int(lp_b.argmax().item())
                pred_b_letter = letters[pred_b_idx]
                c_b = pred_b_idx == correct_idx

            # ─── Grid: 3K passes, K frames each, collect per-pass log-probs ───
            per_axis_preds = {}
            per_axis_correct = {}
            grid_preds = {m: "?" for m in MERGE_FNS}
            grid_correct = {m: False for m in MERGE_FNS}
            c_oracle_axes = False
            c_oracle_all = False
            if run_grid:
                # Precompute ViT cache ONCE for all K² frames (shared across all grid passes)
                cache_3d, tpt = (None, None)
                if not args.no_vit_cache:
                    try:
                        cache_3d, tpt = precompute_visual_cache(
                            model, processor, all_frames, args.max_video_pixels, device)
                    except Exception as e:
                        if args.debug:
                            logger.debug("ViT cache failed, falling back: %s", e)
                        cache_3d, tpt = None, None

                lp_rows = []
                axis_lp_mats = {}
                for axis in axes:
                    idx_groups = AXIS_FNS[axis](K)
                    axis_lps = []
                    for idxs in idx_groups:
                        subset = [all_frames[i] for i in idxs]
                        if cache_3d is not None:
                            lp = score_letters_cached(
                                model, processor, prompt_text, subset, idxs,
                                cache_3d, tpt, variants, device, args.max_video_pixels)
                        else:
                            lp = score_letters_for_pass(
                                model, processor, prompt_text, subset, variants, device,
                                max_video_pixels=args.max_video_pixels)
                        lp_rows.append(lp)
                        axis_lps.append(lp)
                    axis_mat = torch.stack(axis_lps, dim=0)
                    axis_lp_mats[axis] = axis_mat
                    axis_pred_idx = int(axis_mat.mean(dim=0).argmax().item())
                    per_axis_preds[axis] = letters[axis_pred_idx]
                    per_axis_correct[axis] = (axis_pred_idx == correct_idx)

                lp_matrix = torch.stack(lp_rows, dim=0)
                for m, fn in MERGE_FNS.items():
                    merged = fn(lp_matrix)
                    idx = int(merged.argmax().item())
                    grid_preds[m] = letters[idx]
                    grid_correct[m] = (idx == correct_idx)

                # Oracle: best-of per sample
                c_oracle_axes = any(per_axis_correct.values())
                c_oracle_all = c_b or c_oracle_axes or any(grid_correct.values())

            local["baseline"] += int(c_b)
            local["total"] += 1
            for m, ok in grid_correct.items():
                local[f"grid_{m}"] += int(ok)
            for a in axes:
                local[f"axis_{a}"] += int(per_axis_correct.get(a, False))
            local["oracle_axes"] += int(c_oracle_axes)
            local["oracle_all"] += int(c_oracle_all)

            dg = sample["duration_bin"]
            d = local["by_dg"].setdefault(dg, {"baseline": 0, "total": 0})
            d["baseline"] += int(c_b); d["total"] += 1
            for m, ok in grid_correct.items():
                d.setdefault(f"grid_{m}", 0)
                d[f"grid_{m}"] += int(ok)
            for a in axes:
                d.setdefault(f"axis_{a}", 0)
                d[f"axis_{a}"] += int(per_axis_correct.get(a, False))
            d.setdefault("oracle_axes", 0); d["oracle_axes"] += int(c_oracle_axes)
            d.setdefault("oracle_all", 0); d["oracle_all"] += int(c_oracle_all)

            if args.debug:
                tok = processor.tokenizer
                topk_str = ""
                if baseline_topk is not None:
                    topk_str = " base_top5=[" + ", ".join(
                        f"{repr(tok.decode([tid]))}:{lp:.2f}" for tid, lp in baseline_topk) + "]"
                axes_str = " axes={" + ",".join(f"{a}:{p}" for a, p in per_axis_preds.items()) + "}"
                merge_str = " merge={" + ",".join(
                    f"{m}:{grid_preds[m]}{'✓' if grid_correct[m] else '✗'}"
                    for m in MERGE_FNS) + "}"
                logger.info(
                    "[%s] correct=%s  base=%s%s%s%s%s",
                    dg, correct_letter,
                    pred_b_letter, "✓" if c_b else "✗",
                    merge_str, axes_str, topk_str)

        except Exception as e:
            if args.debug:
                import traceback
                logger.warning("Failed: %s\n%s", e, traceback.format_exc())
            continue

    # ── Gather ──
    metric_keys = (["baseline"]
                   + [f"grid_{m}" for m in MERGE_FNS]
                   + [f"axis_{a}" for a in axes]
                   + ["oracle_axes", "oracle_all"])
    if is_dist:
        reduced = {}
        for k in metric_keys + ["total"]:
            t = torch.tensor(local[k], device=device)
            dist.all_reduce(t)
            reduced[k] = t.item()
        total = reduced.pop("total")
        total_correct = reduced

        local_json = json.dumps(local["by_dg"])
        all_jsons = [None] * world_size
        dist.all_gather_object(all_jsons, local_json)
        by_dg = {}
        for j in all_jsons:
            for dg, vals in json.loads(j).items():
                d = by_dg.setdefault(dg, {})
                for k, v in vals.items():
                    d[k] = d.get(k, 0) + v
    else:
        total_correct = {k: local[k] for k in metric_keys}
        total = local["total"]
        by_dg = local["by_dg"]

    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("  GRID SAMPLED ENSEMBLE — %s  (K=%d, axes=%s)",
                    args.benchmark.upper(), K, ",".join(axes))
        logger.info("=" * 80)

        # Table 1: merge strategies
        headers = ["Duration", "Base"] + [m for m in MERGE_FNS] + ["N"]
        logger.info("  " + "  ".join(f"{h:>7}" for h in headers))
        logger.info("  " + "-" * (9 * len(headers)))
        for dg in sorted(by_dg.keys()):
            d = by_dg[dg]
            n = d.get("total", 0)
            if n == 0: continue
            row = [dg, f"{d.get('baseline',0)/n*100:6.1f}"]
            for m in MERGE_FNS:
                row.append(f"{d.get(f'grid_{m}',0)/n*100:6.1f}")
            row.append(str(n))
            logger.info("  " + "  ".join(f"{c:>7}" for c in row))

        if total > 0:
            logger.info("  " + "-" * (9 * len(headers)))
            row = ["OVERALL", f"{total_correct['baseline']/total*100:6.1f}"]
            for m in MERGE_FNS:
                row.append(f"{total_correct[f'grid_{m}']/total*100:6.1f}")
            row.append(str(total))
            logger.info("  " + "  ".join(f"{c:>7}" for c in row))

            best_m = max(MERGE_FNS, key=lambda m: total_correct[f"grid_{m}"])
            best_acc = total_correct[f"grid_{best_m}"] / total * 100
            base_acc = total_correct["baseline"] / total * 100
            logger.info("  BEST merge: %s  %.1f%%  (delta vs baseline: %+.1f%%)",
                        best_m, best_acc, best_acc - base_acc)

        # Table 2: per-axis + oracle
        logger.info("")
        logger.info("  %-10s  %8s  %8s  %8s", "Duration", "Base", "Axes", "All")
        axis_headers = ["Duration", "Base"] + [a for a in axes] + ["OrAxes", "OrAll", "N"]
        logger.info("  " + "  ".join(f"{h:>8}" for h in axis_headers))
        logger.info("  " + "-" * (10 * len(axis_headers)))
        for dg in sorted(by_dg.keys()):
            d = by_dg[dg]
            n = d.get("total", 0)
            if n == 0: continue
            row = [dg, f"{d.get('baseline',0)/n*100:6.1f}"]
            for a in axes:
                row.append(f"{d.get(f'axis_{a}',0)/n*100:6.1f}")
            row.append(f"{d.get('oracle_axes',0)/n*100:6.1f}")
            row.append(f"{d.get('oracle_all',0)/n*100:6.1f}")
            row.append(str(n))
            logger.info("  " + "  ".join(f"{c:>8}" for c in row))
        if total > 0:
            logger.info("  " + "-" * (10 * len(axis_headers)))
            row = ["OVERALL", f"{total_correct['baseline']/total*100:6.1f}"]
            for a in axes:
                row.append(f"{total_correct[f'axis_{a}']/total*100:6.1f}")
            row.append(f"{total_correct['oracle_axes']/total*100:6.1f}")
            row.append(f"{total_correct['oracle_all']/total*100:6.1f}")
            row.append(str(total))
            logger.info("  " + "  ".join(f"{c:>8}" for c in row))

            logger.info("")
            logger.info("  ORACLE(axes)     = %.1f%%  (+%.1f%% over baseline — upper bound if a router picks the right axis)",
                        total_correct["oracle_axes"] / total * 100,
                        (total_correct["oracle_axes"] - total_correct["baseline"]) / total * 100)
            logger.info("  ORACLE(all)      = %.1f%%  (+%.1f%% over baseline — router over {baseline, axes, merges})",
                        total_correct["oracle_all"] / total * 100,
                        (total_correct["oracle_all"] - total_correct["baseline"]) / total * 100)

        with open(args.output, "w") as f:
            json.dump({
                "benchmark": args.benchmark,
                "K": K,
                "axes": axes,
                "baseline_acc": round(total_correct["baseline"] / max(total, 1) * 100, 2),
                "grid_acc": {
                    m: round(total_correct[f"grid_{m}"] / max(total, 1) * 100, 2)
                    for m in MERGE_FNS
                },
                "axis_acc": {
                    a: round(total_correct[f"axis_{a}"] / max(total, 1) * 100, 2)
                    for a in axes
                },
                "oracle_axes_acc": round(total_correct["oracle_axes"] / max(total, 1) * 100, 2),
                "oracle_all_acc": round(total_correct["oracle_all"] / max(total, 1) * 100, 2),
                "by_duration": {
                    dg: {
                        "baseline": round(d.get("baseline", 0) / max(d.get("total", 1), 1) * 100, 2),
                        **{f"grid_{m}": round(d.get(f"grid_{m}", 0) / max(d.get("total", 1), 1) * 100, 2)
                           for m in MERGE_FNS},
                        **{f"axis_{a}": round(d.get(f"axis_{a}", 0) / max(d.get("total", 1), 1) * 100, 2)
                           for a in axes},
                        "oracle_axes": round(d.get("oracle_axes", 0) / max(d.get("total", 1), 1) * 100, 2),
                        "oracle_all": round(d.get("oracle_all", 0) / max(d.get("total", 1), 1) * 100, 2),
                        "n": d.get("total", 0),
                    } for dg, d in by_dg.items()
                },
            }, f, indent=2)
        logger.info("Saved to %s", args.output)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
