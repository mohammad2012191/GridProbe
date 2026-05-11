"""
Two-stage cross-model evaluation: small-VLM selector + large-VLM QA.

Supported configurations:
    --selector_model Qwen/Qwen3-VL-2B-Instruct
    --qa_model       Qwen/Qwen3-VL-30B-A3B-Instruct        (Qwen3-VL family)
    --qa_model       google/gemma-4-26B-A4B-it             (Gemma 4 family)

For each video question we report:
    1. QA-only baseline:      QA model on K² uniformly-sampled frames
    2. QA-only frame-cap:     QA model on its native max frames (e.g. 60 for Gemma 4)
    3. Two-stage (ours):      selector probes K² frames → top-M_eff golden frames
                               passed to QA model for the final answer
    4. Two-stage probe-only:  the selector's own ensemble prediction (sanity check)

Usage:
    # Qwen 2B → Qwen 30B-A3B
    python -m GridProbe.eval.two_stage_eval_crossmodel \
        --data_dir /path/to/Video-MME \
        --selector_model Qwen/Qwen3-VL-2B-Instruct \
        --qa_model       Qwen/Qwen3-VL-30B-A3B-Instruct \
        --K 12 --M auto --pr_gamma 4.0 \
        --max_video_pixels_probe 50176 --max_video_pixels_focus 0 \
        --n_samples 100 --duration_bin long \
        --output ts_crossmodel_qwen30B.json --debug

    # Qwen 2B → Gemma 4 26B-A4B (with frame-cap baseline)
    python -m GridProbe.eval.two_stage_eval_crossmodel \
        --data_dir /path/to/Video-MME \
        --selector_model Qwen/Qwen3-VL-2B-Instruct \
        --qa_model       google/gemma-4-26B-A4B-it \
        --qa_native_max_frames 60 \
        --K 12 --M auto --pr_gamma 4.0 \
        --max_video_pixels_probe 50176 \
        --n_samples 100 --duration_bin long \
        --output ts_crossmodel_gemma4.json --debug
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import gc

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, score_letters_for_pass,
    get_letter_token_id_variants, row_indices, col_indices,
    load_video_mme,
)
from GridProbe.eval.two_stage_eval import (
    stage1_probe, participation_ratio, skewness_selector
)
from GridProbe.eval.video_mme import build_prompt as build_prompt_v1, check_answer as check_answer_v1
from GridProbe.eval.longvideobench import (
    LETTERS_LVB, BINS_LVB, build_prompt_lvb, check_answer_lvb, load_lvb,
)

# v2 utilities
from GridProbe.eval.video_mme_v2 import (
    LETTERS_V2, build_prompt_v2, check_answer_v2, load_video_mme_v2, BINS_V2,
)

logger = logging.getLogger(__name__)
# Module-level LETTERS gets reassigned in main() based on --benchmark
LETTERS = ["A", "B", "C", "D"]
# Module references reassigned for benchmark dispatch
build_prompt = build_prompt_v1
check_answer = check_answer_v1


# ═══════════════════════════════════════════════════════════════
# Backend detection (Qwen3-VL vs Gemma 4)
# ═══════════════════════════════════════════════════════════════

def detect_backend(model_id: str) -> str:
    """Identify which family a model_id belongs to."""
    s = model_id.lower()
    if "gemma" in s:
        return "gemma"
    if "qwen" in s:
        return "qwen"
    raise ValueError(f"Unknown backend for model_id={model_id}; only qwen/gemma supported.")


# ═══════════════════════════════════════════════════════════════
# Loading helpers (Qwen vs Gemma)
# ═══════════════════════════════════════════════════════════════

def load_qa_model(model_id: str, cache_dir: str = None, device_map: str = "balanced",
                   max_memory: dict = None, prefer_dtype=torch.bfloat16):
    """Load QA model + processor, with backend auto-detection.

    Returns (model, processor, backend_name).

    If max_memory is provided (dict {gpu_idx: str like "70GiB"}), it overrides
    device_map's automatic balancing — useful when you want to reserve one GPU
    for the selector model and force the QA model onto the others.
    """
    backend = detect_backend(model_id)
    kw = {"torch_dtype": prefer_dtype}
    if max_memory is not None:
        kw["max_memory"] = max_memory
        kw["device_map"] = "auto"  # required when max_memory is supplied
        logger.info("Loading QA model %s with max_memory=%s",
                    model_id, max_memory)
    else:
        kw["device_map"] = device_map
        logger.info("Loading QA model %s with device_map=%s", model_id, device_map)
    if cache_dir:
        kw["cache_dir"] = cache_dir

    if backend == "qwen":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, cache_dir=os.environ.get("HF_HOME"), **kw)
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"), **kw)
        processor = AutoProcessor.from_pretrained(
            model_id, **({"cache_dir": cache_dir} if cache_dir else {}))
    else:  # gemma
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoModelForMultimodalLM
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_id, cache_dir=os.environ.get("HF_HOME"),  **kw)
        except TypeError:
            model = AutoModelForMultimodalLM.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"), **kw)
        processor = AutoProcessor.from_pretrained(
            model_id, **({"cache_dir": cache_dir} if cache_dir else {}))

    model.eval()

    # Print final layout
    try:
        device_set = sorted({str(p.device) for p in model.parameters()})
        logger.info("QA model device layout: %s", device_set)
    except Exception:
        pass

    return model, processor, backend


def load_selector_model(model_id: str, cache_dir: str = None, device=None,
                         prefer_dtype=torch.bfloat16):
    """Load selector — assumed Qwen3-VL family (used as a small-VLM probe)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = {"torch_dtype": prefer_dtype}
    if cache_dir:
        kw["cache_dir"] = cache_dir
    logger.info("Loading selector model %s ...", model_id)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, cache_dir=os.environ.get("HF_HOME"), **kw)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"), **kw)
    processor = AutoProcessor.from_pretrained(
        model_id, **({"cache_dir": cache_dir} if cache_dir else {}))

    if device is not None:
        model = model.to(device)
    model.eval()
    return model, processor


# ═══════════════════════════════════════════════════════════════
# QA-model letter scoring (backend-aware)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def qa_score_letters_qwen(model, processor, prompt_text, frames,
                           letter_variants, device, max_video_pixels=0):
    """Run a Qwen3-VL forward on `frames`. Returns (letter_lp, n_lm_tokens)."""
    inputs = build_inputs(processor, prompt_text, frames, max_video_pixels)
    n_tokens = int(inputs["input_ids"].shape[1])

    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pvv = inputs.get("pixel_values_videos")
    vgt = inputs.get("video_grid_thw")
    mm_tt = inputs.get("mm_token_type_ids")
    if pvv is not None: pvv = pvv.to(device)
    if vgt is not None: vgt = vgt.to(device)
    if mm_tt is not None: mm_tt = mm_tt.to(device)

    kwargs = {"input_ids": input_ids, "attention_mask": attn, "use_cache": False}
    if pvv is not None: kwargs["pixel_values_videos"] = pvv
    if vgt is not None: kwargs["video_grid_thw"] = vgt
    if mm_tt is not None: kwargs["mm_token_type_ids"] = mm_tt

    out = model(**kwargs)
    logits = out.logits[0, -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    letter_lp = torch.empty(len(letter_variants), dtype=torch.float32)
    for i, ids in enumerate(letter_variants):
        if not ids:
            letter_lp[i] = float("-inf")
        else:
            idx = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
            letter_lp[i] = torch.logsumexp(log_probs[idx], dim=0).cpu()
    return letter_lp, n_tokens


@torch.no_grad()
def qa_score_letters_gemma(model, processor, prompt_text, frames,
                            letter_variants, device, image_token_budget=140):
    """Run a Gemma 4 forward on the (PIL) frames and return letter log-probs.

    Gemma 4's processor accepts interleaved image+text content. We pass each
    frame as a separate image; total context grows with frame count.

    image_token_budget ∈ {70, 140, 280, 560, 1120} — controls per-image tokens.
    Defaults to 140 (suitable for video frames).
    """
    if not frames:
        raise ValueError("Gemma path needs at least one frame")

    # Build content list: image, image, ..., text (per Gemma best-practice ordering).
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    # Gemma's chat template returns a tensor dict directly when tokenize=True
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # Pass image_token_budget if the processor supports it (newer versions only)
    # Capture token count for FLOPs accounting
    if isinstance(inputs, dict) and "input_ids" in inputs:
        n_tokens = int(inputs["input_ids"].shape[1])
    elif hasattr(inputs, "input_ids"):
        n_tokens = int(inputs.input_ids.shape[1])
    else:
        n_tokens = -1
    out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    letter_lp = torch.empty(len(letter_variants), dtype=torch.float32)
    for i, ids in enumerate(letter_variants):
        if not ids:
            letter_lp[i] = float("-inf")
        else:
            idx = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
            letter_lp[i] = torch.logsumexp(log_probs[idx], dim=0).cpu()
    return letter_lp, n_tokens


def qa_answer(model, processor, backend, prompt_text, frames,
               letter_variants, device, max_video_pixels=0):
    """Backend-dispatched single-pass scoring; returns (predicted_letter, confidence, lp, n_tokens)."""
    if backend == "qwen":
        lp, n_tokens = qa_score_letters_qwen(
            model, processor, prompt_text, frames,
            letter_variants, device, max_video_pixels)
    else:  # gemma
        lp, n_tokens = qa_score_letters_gemma(
            model, processor, prompt_text, frames,
            letter_variants, device)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()
    return pred, conf, lp.cpu(), n_tokens


# ═══════════════════════════════════════════════════════════════
# Analytical TFLOPs accounting
# ═══════════════════════════════════════════════════════════════

def infer_lm_dims(model):
    """Pull (d_model, d_ffn, d_kv, n_layers, vocab_size) from model.config.

    Falls back to safe defaults; for MoE models we use the *active* d_ffn so
    the FLOPs estimate reflects per-token compute (which is what dominates).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {"d_model": 2048, "d_ffn": 11008, "d_kv": 2048,
                "n_layers": 28, "vocab_size": 151936}

    # text/lang sub-config
    text_cfg = getattr(cfg, "text_config", cfg)
    d = getattr(text_cfg, "hidden_size", 2048)
    ffn = getattr(text_cfg, "intermediate_size", 4 * d)
    n_h = getattr(text_cfg, "num_attention_heads", 16)
    n_kv = getattr(text_cfg, "num_key_value_heads", n_h)
    n_layers = getattr(text_cfg, "num_hidden_layers", 28)
    vocab = getattr(text_cfg, "vocab_size", getattr(cfg, "vocab_size", 151936))
    d_kv = d * n_kv // max(1, n_h)

    # MoE: use active expert count if exposed.
    n_active = getattr(text_cfg, "num_experts_per_tok", 1)
    n_shared = getattr(text_cfg, "n_shared_experts",
                       getattr(text_cfg, "num_shared_experts", 0))
    moe_ffn = ffn * (n_active + (n_shared if n_shared else 0)) \
              if n_active > 1 else ffn

    return {
        "d_model": d, "d_ffn": moe_ffn, "d_kv": d_kv,
        "n_layers": n_layers, "vocab_size": vocab,
    }


def lm_flops_per_forward(n_tokens, dims):
    """Approximate LM forward FLOPs (multiply-add counted as 2)."""
    if n_tokens is None or n_tokens <= 0:
        return 0
    d = dims["d_model"]; d_ffn = dims["d_ffn"]; d_kv = dims["d_kv"]
    L = dims["n_layers"]; V = dims["vocab_size"]; N = n_tokens
    per_tok_per_layer = (
        2 * (d * d + 2 * d * d_kv)   # QKV proj (with GQA d_kv)
        + 2 * N * d                   # attn scores
        + 2 * N * d                   # attn weighted
        + 2 * d * d                   # out proj
        + 2 * 3 * d * d_ffn           # SwiGLU FFN (up + gate + down)
    )
    return N * L * per_tok_per_layer + 2 * N * d * V


# ═══════════════════════════════════════════════════════════════
# Frame selection helpers
# ═══════════════════════════════════════════════════════════════

def select_uniform(all_frames, n):
    """Pick n uniformly-spaced frames from all_frames."""
    if n >= len(all_frames):
        return list(all_frames)
    idx = np.linspace(0, len(all_frames) - 1, n).round().astype(int).tolist()
    return [all_frames[i] for i in idx]


def select_golden(all_frames, golden_cells, M, K):
    """Pick the M frames corresponding to the top-M golden cells (sorted by importance).

    Returns frames in original temporal order (not importance order) so the LM
    sees them chronologically.
    """
    top_M = golden_cells[:M]
    indices = sorted([r * K + c for r, c in top_M])
    indices = [i for i in indices if i < len(all_frames)]
    return [all_frames[i] for i in indices], indices


# ═══════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════

def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_summary():
    if not torch.cuda.is_available():
        return "no cuda"
    parts = []
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        used_gb = (total - free) / 1024**3
        total_gb = total / 1024**3
        parts.append(f"GPU{d}: {used_gb:.1f}/{total_gb:.1f}GB")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=False, default=None,
                   help="Required for video_mme/video_mme_v2. For lvb, use --lvb_json.")
    p.add_argument("--lvb_json", default=None,
                   help="(lvb only) path to lvb_val.json or a stratified subset JSON.")
    p.add_argument("--lvb_dir", default=None,
                   help="(lvb only) optional override for LVB root dir; defaults to "
                        "the parent of --lvb_json.")
    p.add_argument("--benchmark", choices=["video_mme", "video_mme_v2", "lvb"],
                   default="video_mme",
                   help="video_mme = 4-option (A-D); video_mme_v2 = 8-option (A-H); "
                        "lvb = LongVideoBench (A-E max, with subtitles by default).")
    p.add_argument("--with_subtitle", action="store_true",
                   help="Include subtitles in the prompt. v2: expects "
                        "<data_dir>/subtitles/<vid>.jsonl. lvb: reads subtitle_path "
                        "from each LVB item.")

    # Models
    p.add_argument("--selector_model", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="Small VLM used for grid probing (must be Qwen3-VL family)")
    p.add_argument("--qa_model", required=True,
                   help="Large VLM used for the focused QA pass and baselines")
    p.add_argument("--qa_native_max_frames", type=int, default=None,
                   help="Native frame cap for the QA model; if set, also evaluate "
                        "a 'frame-cap baseline' (uniformly-sampled to that many frames). "
                        "For Gemma 4, set to 60.")

    # Grid hyperparams
    p.add_argument("--K", type=int, default=12, help="K such that K² frames are extracted")
    p.add_argument("--M", default="auto",
                   help="Selection size — int or 'auto' for participation-ratio adaptive")
    p.add_argument("--pr_gamma", type=float, default=4.0)
    p.add_argument("--pr_min_M", type=int, default=2)
    p.add_argument("--pr_max_M", type=int, default=None,
                   help="Cap on M_eff. For Gemma 4 set to qa_native_max_frames.")
    p.add_argument("--max_video_pixels_probe", type=int, default=50176,
                   help="Probe-pass per-frame pixel budget (selector only)")
    p.add_argument("--max_video_pixels_focus", type=int, default=0,
                   help="Focused-pass per-frame pixel budget (QA model on golden frames)")

    # Eval
    p.add_argument("--n_samples", type=int, default=30,
                   help="Cap on samples after sharding (≤0 = all samples in this shard).")
    p.add_argument("--start_index", type=int, default=0,
                   help="Skip the first N samples (post-shard, post-duration_bin filter). "
                        "Use to resume after a crash: rerun with --start_index <last_completed>+1, "
                        "then merge the resulting JSON with the recovered JSON via merge_shards.")
    p.add_argument("--duration_bin", default=None)
    p.add_argument("--n_shards", type=int, default=1,
                   help="Total parallel shards. Each shard processes samples[idx::n_shards].")
    p.add_argument("--shard_index", type=int, default=0,
                   help="0-based shard id (must be < n_shards).")
    p.add_argument("--parallel_gpus", type=int, default=1,
                   help="If > 1, spawn that many child processes (one per GPU) "
                        "with auto-sharding and merge results into --output.")
    p.add_argument("--parallel_stagger_sec", type=int, default=0,
                   help="Seconds to sleep between spawning each parallel worker. "
                        "Mitigates OOM during simultaneous model load.")
    p.add_argument("--parallel_omp_threads", type=int, default=2,
                   help="OMP/MKL thread cap per parallel worker.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--selector_device", default="cuda:0",
                   help="Device for the selector model")
    p.add_argument("--qa_device_map", default="balanced",
                   help="device_map for the QA model (HF accelerate routing). "
                        "Ignored if --qa_skip_selector_gpu is set.")
    p.add_argument("--qa_skip_selector_gpu", action="store_true", default=True,
                   help="Reserve --selector_device exclusively for the selector by "
                        "passing max_memory that excludes that GPU from the QA model. "
                        "Recommended for big QA models on multi-GPU machines (e.g. "
                        "Qwen3-VL-30B) — prevents OOM from selector + QA-share + "
                        "activations all landing on the same GPU.")
    p.add_argument("--qa_per_gpu_gib", type=float, default=None,
                   help="Soft cap on per-GPU memory for the QA model (in GiB). "
                        "Default = (total_gpu_mem - 6 GiB headroom).")
    p.add_argument("--output", default="ts_crossmodel.json")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip_full_baseline", action="store_true",
                   help="Skip the K²-frame baseline (e.g. if QA model can't handle K² frames)")
    args = p.parse_args()

    # ── Real-parallel: if --parallel_gpus N, fan out to N children and exit. ──
    from GridProbe.eval.parallel_launcher import maybe_spawn_parallel_workers
    if maybe_spawn_parallel_workers(
            args, eval_module="GridProbe.eval.two_stage_eval_crossmodel"):
        return

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    K = args.K
    n_frames = K * K
    adaptive_M = (str(args.M).lower() == "auto")
    M_fixed = None if adaptive_M else int(args.M)

    # ── Benchmark dispatch (4-letter v1 vs 8-letter v2) ──
    # Reassign in BOTH this module and the imported two_stage_eval module so that
    # `stage1_probe` (defined there) sees the correct LETTERS list.
    import GridProbe.eval.two_stage_eval as _ts_mod
    global LETTERS, build_prompt, check_answer
    if args.benchmark == "video_mme_v2":
        LETTERS = LETTERS_V2
        build_prompt = build_prompt_v2
        check_answer = check_answer_v2
        _ts_mod.LETTERS = LETTERS_V2
        logger.info("Benchmark: Video-MME-v2 (8-option A-H)")
    elif args.benchmark == "lvb":
        LETTERS = LETTERS_LVB
        build_prompt = build_prompt_lvb
        check_answer = check_answer_lvb
        _ts_mod.LETTERS = LETTERS_LVB
        logger.info("Benchmark: LongVideoBench (5-option A-E max; per-Q candidates dynamic)")
        if not args.lvb_json:
            raise SystemExit("--benchmark lvb requires --lvb_json /path/to/lvb_val.json "
                             "(or to a stratified subset JSON).")
    else:
        LETTERS = ["A", "B", "C", "D"]
        build_prompt = build_prompt_v1
        check_answer = check_answer_v1
        _ts_mod.LETTERS = ["A", "B", "C", "D"]
        logger.info("Benchmark: Video-MME (4-option A-D)")

    # ── Load selector ──
    selector_device = torch.device(args.selector_device if torch.cuda.is_available() else "cpu")
    selector_model, selector_proc = load_selector_model(
        args.selector_model, cache_dir=args.cache_dir, device=selector_device)
    sel_letter_variants = get_letter_token_id_variants(selector_proc.tokenizer, LETTERS)
    logger.info("Selector loaded on %s | %s", selector_device, vram_summary())

    # ── Build max_memory if we want to skip the selector's GPU for the QA model ──
    qa_max_memory = None
    if args.qa_skip_selector_gpu and torch.cuda.is_available():
        # Identify selector GPU index
        try:
            sel_idx = int(str(selector_device).replace("cuda:", ""))
        except Exception:
            sel_idx = 0
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            qa_max_memory = {}
            for i in range(n_gpus):
                if i == sel_idx:
                    qa_max_memory[i] = "0GiB"  # reserved for selector
                else:
                    if args.qa_per_gpu_gib is not None:
                        qa_max_memory[i] = f"{args.qa_per_gpu_gib:.0f}GiB"
                    else:
                        # total - 6GiB headroom for activations / KV cache / fragmentation
                        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        qa_max_memory[i] = f"{int(total_gib - 6)}GiB"
            logger.info("Reserving GPU %d exclusively for the selector; "
                        "routing QA model to GPUs %s",
                        sel_idx, [i for i in range(n_gpus) if i != sel_idx])
        else:
            logger.info("Single GPU detected; --qa_skip_selector_gpu has no effect.")

    # ── Load QA model ──
    qa_model, qa_proc, qa_backend = load_qa_model(
        args.qa_model, cache_dir=args.cache_dir,
        device_map=args.qa_device_map,
        max_memory=qa_max_memory)
    qa_letter_variants = get_letter_token_id_variants(qa_proc.tokenizer, LETTERS)
    qa_device = next(qa_model.parameters()).device
    logger.info("QA model loaded (backend=%s) | %s", qa_backend, vram_summary())

    # Analytical-FLOPs config for both models
    selector_dims = infer_lm_dims(selector_model)
    qa_dims       = infer_lm_dims(qa_model)
    logger.info("Selector LM dims: %s", selector_dims)
    logger.info("QA       LM dims: %s", qa_dims)

    # ── Load samples ──
    if args.benchmark == "video_mme_v2":
        all_samples = load_video_mme_v2(args.data_dir, with_subtitle=args.with_subtitle)
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
    samples = all_samples[:args.n_samples] if args.n_samples > 0 else all_samples
    # ── Skip already-processed samples (resume after crash) ──
    if args.start_index > 0:
        if args.start_index >= len(samples):
            raise SystemExit(
                f"--start_index={args.start_index} is >= total samples "
                f"({len(samples)}); nothing to do.")
        skipped = args.start_index
        samples = samples[args.start_index:]
        logger.info("Resuming from --start_index=%d: skipped %d, %d remaining",
                    args.start_index, skipped, len(samples))
    logger.info("Running %d samples | K=%d | selector=%s | qa=%s (%s)",
                len(samples), K, args.selector_model, args.qa_model, qa_backend)

    # If qa_native_max_frames is set, default the PR cap to that as well.
    if args.qa_native_max_frames is not None and args.pr_max_M is None:
        args.pr_max_M = args.qa_native_max_frames
        logger.info("Setting pr_max_M = qa_native_max_frames = %d", args.pr_max_M)

    # ── Counters: aggregated overall + per bin ──
    # For VMME v1: bin = duration (short/medium/long)
    # For VMME v2: bin = level (1/2/3 — only the 4th question in each group has a level)
    METHODS = ["qa_full", "qa_native", "qa_uniform_M", "probe_only", "two_stage"]
    if args.benchmark == "video_mme_v2":
        BINS = BINS_V2 + ["0"]   # "0" catches questions without an explicit level
    elif args.benchmark == "lvb":
        BINS = BINS_LVB          # ["15","60","600","3600"] — duration_group buckets
    else:
        BINS = ["short", "medium", "long"]

    def fresh_block():
        return {m: 0 for m in METHODS}
    def fresh_timings():
        return {m: [] for m in METHODS + ["selector_probe"]}
    def fresh_flops():
        return {m: [] for m in METHODS}

    correct        = fresh_block()
    correct_by_bin = {b: fresh_block() for b in BINS}
    total          = 0
    total_by_bin   = {b: 0 for b in BINS}

    timings        = fresh_timings()
    timings_by_bin = {b: fresh_timings() for b in BINS}

    flops          = fresh_flops()
    flops_by_bin   = {b: fresh_flops() for b in BINS}

    all_results = []
    adaptive_Ms = []

    for si, sample in enumerate(samples):
        logger.info("[%d/%d] %s [%s]", si + 1, len(samples),
                    sample["id"][:20], sample["duration_bin"])

        frames = extract_frames_uniform(sample["video_path"], n_frames)
        if frames is None:
            logger.warning("  frame extraction failed")
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

        bin_name = sample.get("duration_bin", "?")
        if bin_name not in BINS:
            logger.debug("  skip unknown bin %s", bin_name)
            continue

        try:
            # ════════════════════════════════════════════
            # Stage 1: probe with the small selector
            # ════════════════════════════════════════════
            t0 = time.perf_counter()
            probe = stage1_probe(
                selector_model, selector_proc, prompt_text, frames, K,
                sel_letter_variants, selector_device, args.max_video_pixels_probe)
            t_probe = time.perf_counter() - t0
            timings["selector_probe"].append(t_probe)
            timings_by_bin[bin_name]["selector_probe"].append(t_probe)

            # Selector probe FLOPs: 2K passes through the SELECTOR, each on K frames.
            # The probe-pass token count varies by per-frame budget; we approximate
            # by reading the row_lp tensor shape and doing one extra build_inputs
            # peek to get a representative count.
            try:
                _peek_row_idx = row_indices(K)[0]
                _peek_frames = [frames[i] for i in _peek_row_idx]
                _peek_inp = build_inputs(selector_proc, prompt_text,
                                         _peek_frames, args.max_video_pixels_probe)
                probe_pass_tokens = int(_peek_inp["input_ids"].shape[1])
            except Exception:
                probe_pass_tokens = K * 100  # very rough fallback
            probe_flops_total = 2 * K * lm_flops_per_forward(probe_pass_tokens, selector_dims)

            # Resolve M
            if adaptive_M:
                M = skewness_selector(
                    probe["M_mult"],
                    min_M=args.pr_min_M,
                    max_M=args.pr_max_M)
            else:
                M = M_fixed
            adaptive_Ms.append(M)

            # ════════════════════════════════════════════
            # Stage 2: focused QA pass on golden frames
            # ════════════════════════════════════════════
            golden_frames, golden_indices = select_golden(frames, probe["golden"], M, K)
            t0 = time.perf_counter()
            ts_pred, ts_conf, _, n_tok_2s = qa_answer(
                qa_model, qa_proc, qa_backend, prompt_text, golden_frames,
                qa_letter_variants, qa_device, args.max_video_pixels_focus)
            t_two_stage = time.perf_counter() - t0
            ts_qa_flops = lm_flops_per_forward(n_tok_2s, qa_dims)
            ts_total_flops = probe_flops_total + ts_qa_flops
            timings["two_stage"].append(t_two_stage)
            timings_by_bin[bin_name]["two_stage"].append(t_two_stage)
            flops["two_stage"].append(ts_total_flops)
            flops_by_bin[bin_name]["two_stage"].append(ts_total_flops)

            # ════════════════════════════════════════════
            # Baselines (with the QA model)
            # ════════════════════════════════════════════

            # (i) QA on full K² frames (skip if model can't handle it / user opts out)
            if not args.skip_full_baseline and (
                args.qa_native_max_frames is None or n_frames <= args.qa_native_max_frames
            ):
                t0 = time.perf_counter()
                full_pred, _, _, n_tok_full = qa_answer(
                    qa_model, qa_proc, qa_backend, prompt_text, frames,
                    qa_letter_variants, qa_device, args.max_video_pixels_focus)
                t_full = time.perf_counter() - t0
                full_flops = lm_flops_per_forward(n_tok_full, qa_dims)
                timings["qa_full"].append(t_full)
                timings_by_bin[bin_name]["qa_full"].append(t_full)
                flops["qa_full"].append(full_flops)
                flops_by_bin[bin_name]["qa_full"].append(full_flops)
                c_full = check_answer(full_pred, answer, options)
                correct["qa_full"] += c_full
                correct_by_bin[bin_name]["qa_full"] += c_full
            else:
                full_pred, t_full, c_full, full_flops, n_tok_full = "(skip)", 0.0, 0, 0, 0

            # (ii) QA on native_max frames (the frame-cap baseline; e.g. Gemma 4 at 60)
            if args.qa_native_max_frames is not None:
                native_frames = select_uniform(frames, args.qa_native_max_frames)
                t0 = time.perf_counter()
                native_pred, _, _, n_tok_native = qa_answer(
                    qa_model, qa_proc, qa_backend, prompt_text, native_frames,
                    qa_letter_variants, qa_device, args.max_video_pixels_focus)
                t_native = time.perf_counter() - t0
                native_flops = lm_flops_per_forward(n_tok_native, qa_dims)
                timings["qa_native"].append(t_native)
                timings_by_bin[bin_name]["qa_native"].append(t_native)
                flops["qa_native"].append(native_flops)
                flops_by_bin[bin_name]["qa_native"].append(native_flops)
                c_native = check_answer(native_pred, answer, options)
                correct["qa_native"] += c_native
                correct_by_bin[bin_name]["qa_native"] += c_native
            else:
                native_pred, t_native, c_native, native_flops, n_tok_native = "(n/a)", 0.0, 0, 0, 0

            # (iii) QA on M uniformly-sampled frames (matched-budget control vs two-stage)
            uni_frames = select_uniform(frames, M)
            t0 = time.perf_counter()
            uni_pred, _, _, n_tok_uni = qa_answer(
                qa_model, qa_proc, qa_backend, prompt_text, uni_frames,
                qa_letter_variants, qa_device, args.max_video_pixels_focus)
            t_uni = time.perf_counter() - t0
            uni_flops = lm_flops_per_forward(n_tok_uni, qa_dims)
            timings["qa_uniform_M"].append(t_uni)
            timings_by_bin[bin_name]["qa_uniform_M"].append(t_uni)
            flops["qa_uniform_M"].append(uni_flops)
            flops_by_bin[bin_name]["qa_uniform_M"].append(uni_flops)
            c_uni = check_answer(uni_pred, answer, options)
            correct["qa_uniform_M"] += c_uni
            correct_by_bin[bin_name]["qa_uniform_M"] += c_uni

            # (iv) Selector's own probe ensemble (sanity check on the selector alone)
            c_probe_only = check_answer(probe["ensemble_pred"], answer, options)
            correct["probe_only"] += c_probe_only
            correct_by_bin[bin_name]["probe_only"] += c_probe_only
            timings["probe_only"].append(t_probe)
            timings_by_bin[bin_name]["probe_only"].append(t_probe)
            flops["probe_only"].append(probe_flops_total)
            flops_by_bin[bin_name]["probe_only"].append(probe_flops_total)

            # Two-stage scoring
            c_two_stage = check_answer(ts_pred, answer, options)
            correct["two_stage"] += c_two_stage
            correct_by_bin[bin_name]["two_stage"] += c_two_stage
            total += 1
            total_by_bin[bin_name] += 1

            logger.info(
                "  [%s] ans=%s M=%d | "
                "qa_full=%s(%s,%.1fT) qa_native=%s(%s,%.1fT) qa_uniM=%s(%s,%.1fT) "
                "probe=%s(%s,%.1fT) 2stage=%s(%s,%.1fT) | "
                "t_probe=%.1f t_2s=%.1f t_full=%.1f t_native=%.1f",
                bin_name, answer, M,
                full_pred,   "OK" if c_full else "XX",     full_flops/1e12,
                native_pred, "OK" if c_native else "XX",   native_flops/1e12,
                uni_pred,    "OK" if c_uni else "XX",      uni_flops/1e12,
                probe["ensemble_pred"], "OK" if c_probe_only else "XX", probe_flops_total/1e12,
                ts_pred,     "OK" if c_two_stage else "XX", ts_total_flops/1e12,
                t_probe, t_two_stage, t_full, t_native)

            all_results.append({
                "id": sample["id"],
                "duration_bin": bin_name,
                "question": sample["question"][:100],
                "answer": answer,
                "M_used": M,
                "qa_full":      full_pred,
                "qa_native":    native_pred,
                "qa_uniform_M": uni_pred,
                "probe_only":   probe["ensemble_pred"],
                "two_stage":    ts_pred,
                "two_stage_conf": ts_conf,
                "golden_frames":  golden_indices,
                "row_confs": probe["row_confs"].tolist(),
                "col_confs": probe["col_confs"].tolist(),
                # latencies
                "t_probe":     t_probe,
                "t_two_stage": t_two_stage,
                "t_qa_full":   t_full,
                "t_qa_native": t_native,
                "t_qa_uniform_M": t_uni,
                # token counts (LM seq lengths)
                "n_tok_two_stage": n_tok_2s,
                "n_tok_qa_full":   n_tok_full,
                "n_tok_qa_native": n_tok_native,
                "n_tok_qa_uniform_M": n_tok_uni,
                "n_tok_probe_per_pass": probe_pass_tokens,
                # flops (analytical, in raw units)
                "flops_two_stage": ts_total_flops,
                "flops_qa_full":   full_flops,
                "flops_qa_native": native_flops,
                "flops_qa_uniform_M": uni_flops,
                "flops_probe":     probe_flops_total,
            })

            free_cuda()

        except torch.cuda.OutOfMemoryError as e:
            logger.warning("  OOM: %s", e)
            free_cuda()
            continue
        except Exception as e:
            logger.warning("  failed: %s", e)
            if args.debug:
                import traceback; traceback.print_exc()
            continue

    # ════════════════════════════════════════════
    # Summary — per-bin and overall, with TFLOPs
    # ════════════════════════════════════════════
    method_labels = {
        "qa_full":      "QA on K² frames (baseline)",
        "qa_native":    f"QA on native-{args.qa_native_max_frames} (uniform)" if args.qa_native_max_frames else "QA on native frames",
        "qa_uniform_M": "QA on M uniform (matched budget)",
        "probe_only":   "Selector probe ensemble alone",
        "two_stage":    f"TWO-STAGE selector→QA (M={'auto' if adaptive_M else M_fixed})",
    }

    def fmt_block(n, c_block, t_block, f_block, header):
        if n <= 0:
            print(f"  ({header}: no samples)")
            return
        print(f"  ── {header}  (n={n}) ──")
        print(f"  {'Method':<42} {'Acc':>7} {'Correct':>10} {'Avg time':>10} {'Avg TFLOPs':>11}")
        print("  " + "-" * 86)
        for k in METHODS:
            if k == "qa_native" and args.qa_native_max_frames is None:
                continue
            if k == "qa_full" and not t_block.get("qa_full"):
                continue
            if not t_block.get(k):
                # method may be valid but no samples for this bin
                if c_block.get(k, 0) == 0:
                    continue
            acc = c_block[k] / n * 100 if n else 0.0
            t_list = t_block.get(k, [])
            f_list = f_block.get(k, [])
            avg_t = float(np.mean(t_list)) if t_list else 0.0
            avg_f = float(np.mean(f_list)) if f_list else 0.0
            label = method_labels[k]
            print(f"  {label:<42} {acc:>6.1f}%  {c_block[k]:>4}/{n:<4} {avg_t:>9.2f}s {avg_f/1e12:>10.2f}T")
        print()

    print("\n" + "=" * 92)
    print(f"  CROSS-MODEL EVAL  selector={args.selector_model}  →  qa={args.qa_model}")
    print(f"  K={K} | M={'auto' if adaptive_M else M_fixed} | "
          f"probe_px={args.max_video_pixels_probe} | focus_px={args.max_video_pixels_focus}")
    if args.qa_native_max_frames:
        print(f"  qa_native_max_frames={args.qa_native_max_frames}")
    print("=" * 92)

    # Per-bin blocks
    for bn in BINS:
        fmt_block(total_by_bin[bn], correct_by_bin[bn],
                  timings_by_bin[bn], flops_by_bin[bn],
                  bn.upper())

    # Overall block
    fmt_block(total, correct, timings, flops, "OVERALL")

    # Headline deltas vs the most relevant baseline(s)
    if total > 0:
        ts_acc = correct["two_stage"] / total * 100
        if args.qa_native_max_frames is not None and timings["qa_native"]:
            nat_acc = correct["qa_native"] / total * 100
            print(f"  Δ TWO-STAGE vs QA-native-{args.qa_native_max_frames}:  "
                  f"{ts_acc - nat_acc:+.1f}% (overall)")
            for bn in BINS:
                if total_by_bin[bn] > 0 and timings_by_bin[bn].get("qa_native"):
                    ts_b   = correct_by_bin[bn]["two_stage"] / total_by_bin[bn] * 100
                    nat_b  = correct_by_bin[bn]["qa_native"] / total_by_bin[bn] * 100
                    print(f"      {bn:>7}: {ts_b - nat_b:+.1f}%")
        if timings["qa_full"]:
            full_acc = correct["qa_full"] / total * 100
            print(f"  Δ TWO-STAGE vs QA-full (K²={n_frames}):  "
                  f"{ts_acc - full_acc:+.1f}% (overall)")
            for bn in BINS:
                if total_by_bin[bn] > 0 and timings_by_bin[bn].get("qa_full"):
                    ts_b   = correct_by_bin[bn]["two_stage"] / total_by_bin[bn] * 100
                    full_b = correct_by_bin[bn]["qa_full"] / total_by_bin[bn] * 100
                    print(f"      {bn:>7}: {ts_b - full_b:+.1f}%")

        # Average compute ratio (two-stage vs each baseline) on overall
        def safe_ratio(num, den):
            return num / den if den > 0 else float("nan")
        for ref in ["qa_full", "qa_native", "qa_uniform_M"]:
            if not flops[ref]:
                continue
            avg_ts = float(np.mean(flops["two_stage"]))
            avg_ref = float(np.mean(flops[ref]))
            print(f"  TFLOPs ratio (two_stage / {ref}):  {safe_ratio(avg_ts, avg_ref):.2f}×")

        if adaptive_M and adaptive_Ms:
            Ms = np.array(adaptive_Ms)
            print(f"\n  Adaptive M stats: min={Ms.min()} median={int(np.median(Ms))} "
                  f"mean={Ms.mean():.1f} max={Ms.max()}")
    print("=" * 92)

    with open(args.output, "w") as f:
        json.dump({
            "selector_model": args.selector_model,
            "qa_model": args.qa_model,
            "qa_backend": qa_backend,
            "K": K, "M_mode": "auto" if adaptive_M else M_fixed,
            "pr_gamma": args.pr_gamma,
            "qa_native_max_frames": args.qa_native_max_frames,
            "max_video_pixels_probe": args.max_video_pixels_probe,
            "max_video_pixels_focus": args.max_video_pixels_focus,
            "selector_dims": selector_dims,
            "qa_dims": qa_dims,
            "total": total,
            "total_by_bin": total_by_bin,
            "correct": correct,
            "correct_by_bin": correct_by_bin,
            "avg_time_overall":  {k: (float(np.mean(timings[k])) if timings[k] else 0.0)
                                   for k in METHODS + ["selector_probe"]},
            "avg_time_by_bin":   {b: {k: (float(np.mean(timings_by_bin[b][k]))
                                          if timings_by_bin[b][k] else 0.0)
                                       for k in METHODS + ["selector_probe"]} for b in BINS},
            "avg_tflops_overall":{k: (float(np.mean(flops[k]))/1e12 if flops[k] else 0.0)
                                   for k in METHODS},
            "avg_tflops_by_bin": {b: {k: (float(np.mean(flops_by_bin[b][k]))/1e12
                                          if flops_by_bin[b][k] else 0.0)
                                       for k in METHODS} for b in BINS},
            "adaptive_Ms": adaptive_Ms,
            "results": all_results,
        }, f, indent=2)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
