"""
Grid-Collated Image Variant
============================
After GridProbe selects M golden frames, we collate them into a single high-
resolution image arranged as a sqrt(M)×sqrt(M) grid (sorted by time), and feed
that ONE image to the VLM instead of M separate frames.

Hypothesis: VLMs spend a roughly per-image fixed budget. M separate images
incur M×per-image overhead; one collated grid image preserves all M frames in
a single visual context at the cost of per-frame resolution.

Supports:
  - The full GridProbe pipeline (probe → importance map → adaptive M)
  - All M-selectors (--selector pr|skew|mean|otsu, plus --M <int>)
  - Both Video-MME (4-letter) and Video-MME-v2 (8-letter)
  - Optional side-by-side: regular two-stage vs collated

Usage:
    python -m GridProbe.eval.grid_image_collated \
        --data_dir /path/to/Video-MME-v2 \
        --benchmark video_mme_v2 \
        --K 12 --M auto --selector skew \
        --collated_size 1024 \
        --max_video_pixels_probe 50176 \
        --n_samples 200 \
        --also_run_normal_2stage \
        --output ts_v2_collated.json --debug
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Pull from existing pipeline; reassign module-level LETTERS as needed.
import GridProbe.eval.two_stage_eval as _ts_mod
from GridProbe.eval.two_stage_eval import (
    stage1_probe, select_M, run_baseline_full, stage2_focused,
)
from GridProbe.eval.two_stage_eval_crossmodel import (
    infer_lm_dims, lm_flops_per_forward,
)
from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, get_letter_token_id_variants,
    row_indices,
)
from GridProbe.eval.video_mme_v2 import (
    LETTERS_V2, build_prompt_v2, check_answer_v2, load_video_mme_v2, BINS_V2,
)
from GridProbe.eval.grid_sampled_ensemble_eval import load_video_mme
from GridProbe.eval.video_mme import build_prompt as build_prompt_v1, check_answer as check_answer_v1

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Grid-collated image construction
# ═══════════════════════════════════════════════════════════════

def collate_frames_to_grid(frames, target_side: int = 1024) -> Image.Image:
    """Arrange `frames` as a sqrt(M)×sqrt(M) grid in one PIL image, time-ordered.

    `target_side` controls the COMBINED output side length (in pixels). Per-cell
    side = target_side / grid_k. If M is not a perfect square we pad the grid
    with the LAST frame so the grid stays rectangular.

    Args:
        frames: list of PIL.Image, already in temporal order (chronological)
        target_side: final canvas side length in pixels (e.g. 1024)

    Returns: a single PIL.Image of size (target_side, target_side).
    """
    M = len(frames)
    if M == 0:
        raise ValueError("Need at least 1 frame to collate")
    grid_k = int(math.ceil(math.sqrt(M)))
    cell_side = max(1, target_side // grid_k)
    canvas_side = cell_side * grid_k

    canvas = Image.new("RGB", (canvas_side, canvas_side), (0, 0, 0))
    last = frames[-1]
    for i in range(grid_k * grid_k):
        f = frames[i] if i < M else last
        thumb = f.convert("RGB").resize((cell_side, cell_side), Image.BILINEAR)
        r, c = divmod(i, grid_k)
        canvas.paste(thumb, (c * cell_side, r * cell_side))
    if canvas.size[0] != target_side:
        canvas = canvas.resize((target_side, target_side), Image.BILINEAR)
    return canvas


# ═══════════════════════════════════════════════════════════════
# Image-only letter scoring (single image, not video)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def score_letters_single_image(model, processor, prompt_text, pil_image,
                                letter_variants, device, max_pixels: int = 0):
    """Run Qwen3-VL on ONE image (not a video). Returns (letter_lp, n_lm_tokens).

    NOTE: we do NOT use qwen_vl_utils.process_vision_info here — that function
    is for videos and (depending on the installed version) returns 2/3/4 values.
    For a single image we go straight through the processor.
    """
    image_item = {"type": "image", "image": pil_image}
    if max_pixels and max_pixels > 0:
        image_item["max_pixels"] = int(max_pixels)
    messages = [{"role": "user", "content": [
        image_item, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image],
                       return_tensors="pt", padding=True)

    n_tok = int(inputs["input_ids"].shape[1])
    input_ids = inputs["input_ids"].to(device)
    attn      = inputs["attention_mask"].to(device)
    pv        = inputs.get("pixel_values")
    igt       = inputs.get("image_grid_thw")
    mm_tt     = inputs.get("mm_token_type_ids")
    if pv is not None:    pv    = pv.to(device)
    if igt is not None:   igt   = igt.to(device)
    if mm_tt is not None: mm_tt = mm_tt.to(device)

    kw = {"input_ids": input_ids, "attention_mask": attn, "use_cache": False}
    if pv  is not None: kw["pixel_values"]   = pv
    if igt is not None: kw["image_grid_thw"] = igt
    if mm_tt is not None: kw["mm_token_type_ids"] = mm_tt

    out = model(**kw)
    logits = out.logits[0, -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    letter_lp = torch.empty(len(letter_variants), dtype=torch.float32)
    for i, ids in enumerate(letter_variants):
        if not ids:
            letter_lp[i] = float("-inf")
        else:
            idx = torch.tensor(ids, device=log_probs.device, dtype=torch.long)
            letter_lp[i] = torch.logsumexp(log_probs[idx], dim=0).cpu()
    return letter_lp, n_tok


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def select_golden_frames(all_frames, golden_cells, M, K):
    """Pick top-M cells by importance, return frames in TEMPORAL order."""
    top_M = golden_cells[:M]
    indices = sorted([r * K + c for r, c in top_M
                       if (r * K + c) < len(all_frames)])
    return [all_frames[i] for i in indices], indices


def _peek_probe_tokens(processor, prompt, frames, K, probe_pixels):
    """One peek to get a probe-pass token count for FLOPs accounting."""
    try:
        inp = build_inputs(processor, prompt,
                            [frames[i] for i in row_indices(K)[0]],
                            probe_pixels)
        return int(inp["input_ids"].shape[1])
    except Exception:
        return K * 100


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--benchmark", choices=["video_mme", "video_mme_v2"],
                   default="video_mme_v2")
    p.add_argument("--with_subtitle", action="store_true")

    p.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")

    p.add_argument("--K", type=int, default=12)
    p.add_argument("--M", default="auto",
                   help="int (fixed) or 'auto' (adaptive via --selector)")
    p.add_argument("--selector", default="skew", choices=["pr", "skew", "mean", "otsu"])
    p.add_argument("--pr_gamma", type=float, default=4.0)
    p.add_argument("--pr_min_M", type=int, default=2)
    p.add_argument("--pr_max_M", type=int, default=None)
    p.add_argument("--pr_subtract_min", action="store_true", default=True)
    p.add_argument("--pr_mass", type=float, default=0.9)
    p.add_argument("--skew_scale", type=float, default=3.0)

    p.add_argument("--max_video_pixels_probe", type=int, default=50176)
    p.add_argument("--max_video_pixels_focus", type=int, default=0,
                   help="Used for the K²-frame baseline AND the regular two-stage path "
                        "(if --also_run_normal_2stage).")

    # Collated-image-specific
    p.add_argument("--collated_size", type=int, default=1024,
                   help="Total side length (px) of the collated grid image.")
    p.add_argument("--collated_max_pixels", type=int, default=0,
                   help="Optional max_pixels hint for the processor on the collated image.")
    p.add_argument("--also_run_normal_2stage", action="store_true",
                   help="Also run the regular M-frame two-stage path for comparison.")
    p.add_argument("--skip_full_baseline", action="store_true",
                   help="Skip the K²-frame baseline forward. Useful when iterating on the "
                        "collated path and the baseline numbers are already known from "
                        "an earlier two_stage_eval run on the same K. Saves the largest "
                        "single forward per sample.")

    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--duration_bin", default=None)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default="ts_collated.json")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Benchmark dispatch (the key fix to avoid LETTERS[i] index errors) ──
    if args.benchmark == "video_mme_v2":
        LETTERS_LOCAL = LETTERS_V2
        build_prompt  = build_prompt_v2
        check_answer  = check_answer_v2
        loader        = lambda d: load_video_mme_v2(d, with_subtitle=args.with_subtitle)
        BINS = BINS_V2 + ["0"]
        logger.info("Benchmark: Video-MME-v2 (8 options)")
    else:
        LETTERS_LOCAL = ["A", "B", "C", "D"]
        build_prompt  = build_prompt_v1
        check_answer  = check_answer_v1
        loader        = load_video_mme
        BINS = ["short", "medium", "long"]
        logger.info("Benchmark: Video-MME (4 options)")

    # CRITICAL: reassign LETTERS in the imported module so stage1_probe / stage2_focused see it.
    _ts_mod.LETTERS = LETTERS_LOCAL

    K = args.K
    n_frames = K * K
    adaptive_M = (str(args.M).lower() == "auto")
    M_fixed = None if adaptive_M else int(args.M)

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
    letter_variants = get_letter_token_id_variants(processor.tokenizer, LETTERS_LOCAL)
    model_dims = infer_lm_dims(model)
    logger.info("LM dims: %s", model_dims)
    logger.info("Letters in use (n=%d): %s", len(LETTERS_LOCAL), LETTERS_LOCAL)

    # ── Load samples ──
    all_samples = loader(args.data_dir)
    if args.duration_bin:
        all_samples = [s for s in all_samples if s["duration_bin"] == args.duration_bin]
    if args.n_shards > 1:
        all_samples = all_samples[args.shard_index::args.n_shards]
        logger.info("Shard %d/%d → %d samples", args.shard_index, args.n_shards,
                    len(all_samples))
    samples = all_samples[:args.n_samples] if args.n_samples > 0 else all_samples
    logger.info("Running %d samples | K=%d | M=%s/%s | collated=%dpx",
                len(samples), K,
                "auto" if adaptive_M else M_fixed, args.selector, args.collated_size)

    # ── Counters ──
    METHODS = ["collated"]
    if args.also_run_normal_2stage:
        METHODS.append("two_stage")
    if not args.skip_full_baseline:
        METHODS.append("baseline")

    correct        = {m: 0  for m in METHODS}
    correct_by_bin = {b: {m: 0  for m in METHODS} for b in BINS}
    total          = 0
    total_by_bin   = {b: 0 for b in BINS}
    timings        = {m: [] for m in METHODS}
    timings_by_bin = {b: {m: [] for m in METHODS} for b in BINS}
    flops          = {m: [] for m in METHODS}
    flops_by_bin   = {b: {m: [] for m in METHODS} for b in BINS}
    adaptive_Ms    = []
    all_results    = []

    for si, sample in enumerate(samples):
        logger.info("[%d/%d] %s", si + 1, len(samples), str(sample.get("id", ""))[:20])

        frames = extract_frames_uniform(sample["video_path"], n_frames)
        if frames is None:
            logger.warning("  frame extraction failed")
            continue

        if args.benchmark == "video_mme_v2":
            prompt = build_prompt(
                sample["question"], sample["options"],
                subtitle=sample.get("subtitle") if args.with_subtitle else None)
        else:
            prompt = build_prompt(sample["question"], sample["options"])

        answer  = sample["answer"]
        options = sample.get("options", [])
        bin_name = sample.get("duration_bin", "?")
        if bin_name not in BINS:
            bin_name = None

        try:
            # ──── Stage 1: probe ────
            t0 = time.perf_counter()
            probe = stage1_probe(model, processor, prompt, frames, K,
                                  letter_variants, device, args.max_video_pixels_probe)
            t_probe = time.perf_counter() - t0

            # ──── Resolve M (fixed or adaptive via --selector) ────
            if adaptive_M:
                M = select_M(args.selector, probe["M_mult"], args)
            else:
                M = M_fixed
            adaptive_Ms.append(M)

            # ──── Pick golden frames in temporal order ────
            golden_frames, golden_idx = select_golden_frames(
                frames, probe["golden"], M, K)

            # Pre-compute probe-pass token count once (for FLOPs accounting)
            probe_pass_tokens = _peek_probe_tokens(
                processor, prompt, frames, K, args.max_video_pixels_probe)
            f_probe = 2 * K * lm_flops_per_forward(probe_pass_tokens, model_dims)

            # ──── Collated single-image pass ────
            t0 = time.perf_counter()
            collated_img = collate_frames_to_grid(golden_frames, args.collated_size)
            lp_c, n_tok_c = score_letters_single_image(
                model, processor, prompt, collated_img,
                letter_variants, device, args.collated_max_pixels)
            t_collated = time.perf_counter() - t0
            pred_c = LETTERS_LOCAL[lp_c.argmax().item()]
            c_collated = check_answer(pred_c, answer, options)
            f_collated = f_probe + lm_flops_per_forward(n_tok_c, model_dims)

            # ──── (Optional) regular two-stage on M frames as separate images ────
            pred_ts, c_ts, t_ts, f_ts, n_tok_ts = "(skip)", 0, 0.0, 0, 0
            if args.also_run_normal_2stage:
                t0 = time.perf_counter()
                focused = stage2_focused(
                    model, processor, prompt, frames,
                    probe["golden"], M, K,
                    letter_variants, device, args.max_video_pixels_focus)
                t_ts = time.perf_counter() - t0
                pred_ts = focused["pred"]
                c_ts = check_answer(pred_ts, answer, options)
                try:
                    _peek_2s = build_inputs(processor, prompt,
                                            [frames[i] for i in golden_idx],
                                            args.max_video_pixels_focus)
                    n_tok_ts = int(_peek_2s["input_ids"].shape[1])
                except Exception:
                    n_tok_ts = -1
                f_ts = f_probe + lm_flops_per_forward(n_tok_ts, model_dims)

            # ──── Baseline (full K² frames) ────
            if args.skip_full_baseline:
                pred_b, c_b, t_base, n_tok_b, f_b = "(skip)", 0, 0.0, 0, 0
            else:
                t0 = time.perf_counter()
                base_full = run_baseline_full(
                    model, processor, prompt, frames,
                    letter_variants, device, args.max_video_pixels_focus)
                t_base = time.perf_counter() - t0
                pred_b = base_full["pred"]
                c_b = check_answer(pred_b, answer, options)
                try:
                    _peek_full = build_inputs(processor, prompt, frames,
                                              args.max_video_pixels_focus)
                    n_tok_b = int(_peek_full["input_ids"].shape[1])
                except Exception:
                    n_tok_b = -1
                f_b = lm_flops_per_forward(n_tok_b, model_dims)

            # ──── Tally ────
            tally_list = [("collated", c_collated, t_collated, f_collated)]
            if not args.skip_full_baseline:
                tally_list.append(("baseline", c_b, t_base, f_b))
            if args.also_run_normal_2stage:
                tally_list.append(("two_stage", c_ts, t_ts, f_ts))
            for m, c, t, f in tally_list:
                correct[m] += c
                timings[m].append(t)
                flops[m].append(f)
                if bin_name is not None:
                    correct_by_bin[bin_name][m] += c
                    timings_by_bin[bin_name][m].append(t)
                    flops_by_bin[bin_name][m].append(f)
            total += 1
            if bin_name is not None:
                total_by_bin[bin_name] += 1

            log_extra = ""
            if args.also_run_normal_2stage:
                log_extra = f"  2stage={pred_ts}({'OK' if c_ts else 'XX'},{f_ts/1e12:.1f}T)"
            logger.info(
                "  ans=%s M=%d gridK=%d | base=%s(%s,%.1fT)  collated=%s(%s,%.1fT)%s "
                "| t_probe=%.1f t_coll=%.1f t_base=%.1f",
                answer, M, int(math.ceil(math.sqrt(M))),
                pred_b, "OK" if c_b else "XX", f_b/1e12,
                pred_c, "OK" if c_collated else "XX", f_collated/1e12,
                log_extra, t_probe, t_collated, t_base)

            all_results.append({
                "id": sample.get("id", ""),
                "duration_bin": sample.get("duration_bin", ""),
                "answer": answer,
                "M_used": M,
                "grid_K_collated": int(math.ceil(math.sqrt(M))),
                "collated_pred":  pred_c,
                "two_stage_pred": pred_ts,
                "baseline_pred":  pred_b,
                "n_tok_collated":  n_tok_c,
                "n_tok_two_stage": n_tok_ts,
                "n_tok_baseline":  n_tok_b,
                "flops_collated":  f_collated,
                "flops_two_stage": f_ts,
                "flops_baseline":  f_b,
                "flops_probe":     f_probe,
                "t_probe":    t_probe,
                "t_collated": t_collated,
                "t_two_stage": t_ts,
                "t_baseline": t_base,
                "golden_frames": golden_idx,
            })

        except Exception as e:
            logger.warning("  failed: %s", e)
            if args.debug:
                import traceback; traceback.print_exc()
            continue

    # ── Summary ──
    print("\n" + "=" * 92)
    print(f"  GRID-COLLATED IMAGE EVAL  K={K}  M={'auto' if adaptive_M else M_fixed}/{args.selector}  "
          f"collated_side={args.collated_size}  n={total}")
    print("=" * 92)

    def fmt(n, c_block, t_block, f_block, header):
        if n <= 0:
            print(f"  ({header}: no samples)")
            return
        print(f"  ── {header}  (n={n}) ──")
        print(f"  {'Method':<22} {'Acc':>7} {'Correct':>10} {'Avg time':>10} {'Avg TFLOPs':>11}")
        print("  " + "-" * 70)
        for m in METHODS:
            acc = c_block[m] / n * 100
            avg_t = float(np.mean(t_block[m])) if t_block[m] else 0.0
            avg_f = float(np.mean(f_block[m])) if f_block[m] else 0.0
            print(f"  {m:<22} {acc:>6.1f}%  {c_block[m]:>4}/{n:<4} "
                  f"{avg_t:>9.2f}s {avg_f/1e12:>10.2f}T")
        print()

    for bn in BINS:
        fmt(total_by_bin[bn], correct_by_bin[bn], timings_by_bin[bn],
            flops_by_bin[bn], f"BIN {bn}")
    fmt(total, correct, timings, flops, "OVERALL")

    if total > 0 and adaptive_Ms:
        Ms = np.array(adaptive_Ms)
        print(f"  Adaptive M: min={Ms.min()} median={int(np.median(Ms))} "
              f"mean={Ms.mean():.1f} max={Ms.max()}")
    print("=" * 92)

    with open(args.output, "w") as fp:
        json.dump({
            "benchmark": args.benchmark,
            "K": K, "n_frames": n_frames,
            "M_mode": ("auto/" + args.selector) if adaptive_M else M_fixed,
            "selector": args.selector,
            "collated_size": args.collated_size,
            "model_dims": model_dims,
            "total": total, "total_by_bin": total_by_bin,
            "correct": correct, "correct_by_bin": correct_by_bin,
            "avg_time_overall":  {m: (float(np.mean(timings[m])) if timings[m] else 0.0)
                                   for m in METHODS},
            "avg_tflops_overall":{m: (float(np.mean(flops[m]))/1e12 if flops[m] else 0.0)
                                   for m in METHODS},
            "adaptive_Ms": adaptive_Ms,
            "results": all_results,
        }, fp, indent=2)
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
