"""
MDP3 Frame Selection Baseline
=============================
Run MDP3 (SigLIP scoring + listwise DPP selection) as the selector, then
forward the selected frames through the same QA VLM we use for GridProbe.
Logs preds, frame indices, per-stage timings, token counts, and analytical
FLOPs in the same JSON schema as `two_stage_eval.py`, so the result file
can drop straight into existing analysis tooling (analyze_results.py,
plot_pareto_*.py).

Usage (V2):
    python -m GridProbe.eval.mdp3_eval \
        --benchmark video_mme_v2 --data_dir /path/to/Video-MME \
        --vlm_model Qwen/Qwen3-VL-2B-Instruct \
        --K 12 --M 8 --n_samples 100 \
        --output mdp3_v2_2B_K12_M8.json

Usage (LVB):
    python -m GridProbe.eval.mdp3_eval \
        --benchmark lvb --lvb_json /path/to/lvb_val.json --with_subtitle \
        --vlm_model Qwen/Qwen3-VL-2B-Instruct \
        --K 12 --M 8 --n_samples 100 \
        --output mdp3_lvb_2B_K12_M8.json

Output JSON schema (per-sample fields, all parallel to two_stage_eval.py):
    id, duration_bin, question, answer, M_used,
    mdp3                       — MDP3 prediction (single letter)
    baseline_full              — full-pool baseline pred (None if --skip_full_baseline)
    mdp3_selected_frames       — list of M selected frame indices into the K^2 pool
    t_mdp3_selector            — wall time for SigLIP + DPP
    t_mdp3_qa                  — wall time for the QA VLM forward on selected frames
    t_baseline                 — wall time for the K^2 full baseline (if not skipped)
    n_tok_mdp3_qa, n_tok_baseline_full
    flops_mdp3_selector        — analytical (SigLIP image+text passes; see mdp3_selector.py)
    flops_mdp3_qa              — analytical (QA forward on selected frames)
    flops_mdp3_total           — selector + QA
    flops_baseline_full
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, score_letters_for_pass,
    get_letter_token_id_variants, load_video_mme,
)
from GridProbe.eval.video_mme_v2 import (
    LETTERS_V2, build_prompt_v2, check_answer_v2, load_video_mme_v2, BINS_V2,
)
from GridProbe.eval.longvideobench import (
    LETTERS_LVB, BINS_LVB, build_prompt_lvb, check_answer_lvb, load_lvb,
)
from GridProbe.eval.two_stage_eval_crossmodel import (
    infer_lm_dims, lm_flops_per_forward,
)
from GridProbe.eval.mdp3_selector import (
    MDP3, estimate_mdp3_selector_flops,
)

logger = logging.getLogger(__name__)

# Module-level constants — reassigned by main() based on --benchmark.
# V1 (4-option Video-MME) is loaded lazily inside main() to avoid a hard
# dependency on the GridProbe.eval.video_mme module when only running V2 or LVB.
LETTERS = ["A", "B", "C", "D"]
build_prompt = None
check_answer = None


# ═══════════════════════════════════════════════════════════════
# MDP3 selection + focused QA pass
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def mdp3_focused(qa_model, qa_processor, prompt_text, all_frames,
                 selected_indices, letter_variants, device,
                 max_video_pixels_focus):
    """Run ONE QA forward on the MDP3-selected frames, sorted by time."""
    selected_indices = sorted(int(i) for i in selected_indices)
    selected_frames = [all_frames[i] for i in selected_indices]

    lp = score_letters_for_pass(
        qa_model, qa_processor, prompt_text, selected_frames,
        letter_variants, device, max_video_pixels_focus)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()

    return {
        "pred": pred,
        "conf": conf,
        "frame_indices": selected_indices,
        "lp": lp.cpu(),
    }


@torch.no_grad()
def run_baseline_full(qa_model, qa_processor, prompt_text, all_frames,
                      letter_variants, device, max_video_pixels):
    """Standard baseline: one pass on all K^2 frames."""
    lp = score_letters_for_pass(
        qa_model, qa_processor, prompt_text, all_frames,
        letter_variants, device, max_video_pixels)
    pred = LETTERS[lp.argmax().item()]
    conf = F.softmax(lp.cpu(), dim=-1).max().item()
    return {"pred": pred, "conf": conf, "lp": lp.cpu()}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=False, default=None,
                   help="Required for video_mme/video_mme_v2.")
    p.add_argument("--benchmark", choices=["video_mme", "video_mme_v2", "lvb"],
                   default="video_mme_v2")
    p.add_argument("--lvb_json", default=None,
                   help="(lvb only) path to lvb_val.json or stratified subset.")
    p.add_argument("--lvb_dir", default=None,
                   help="(lvb only) override LVB root dir.")
    p.add_argument("--with_subtitle", action="store_true")
    p.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="QA VLM (the model that produces the final answer).")
    p.add_argument("--siglip_model", default="google/siglip-so400m-patch14-384",
                   help="SigLIP backbone for MDP3 scoring.")
    p.add_argument("--K", type=int, default=12,
                   help="Frame pool side. Pool size = K^2.")
    p.add_argument("--M", type=int, default=8,
                   help="Number of frames MDP3 selects (paper default = 8). "
                        "Used as fallback when --per_sample_M_from is set but "
                        "a sample id is missing from the lookup.")
    p.add_argument("--per_sample_M_from", default=None,
                   help="(Matched-compute mode.) Path to a GridProbe results "
                        "JSON (e.g., from two_stage_eval.py). For each sample "
                        "id, MDP3 will set n_selection = that sample's "
                        "M_used (the per-question adaptive budget GridProbe "
                        "chose via the σ statistic). Falls back to --M when "
                        "an id is missing. This is the cleanest selector-vs-"
                        "selector comparison: same per-question budget, only "
                        "selection method differs.")
    p.add_argument("--mdp3_lambda", type=float, default=0.2,
                   help="MDP3 relevance/diversity balance (paper default = 0.2).")
    p.add_argument("--max_video_pixels_focus", type=int, default=0,
                   help="Resolution for the QA forward (0 = full).")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--duration_bin", default=None)
    p.add_argument("--filter_ids", default=None,
                   help="Comma-separated list of sample IDs to run on. "
                        "Useful for spot-checking specific qualitative examples "
                        "(e.g., --filter_ids '039-4,507-1,326-1'). All other "
                        "samples are skipped.")
    p.add_argument("--filter_ids_file", default=None,
                   help="Path to a file with one sample ID per line. Same "
                        "effect as --filter_ids but avoids huge shell args. "
                        "If both are given, the union is used.")
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default="mdp3_eval.json")
    p.add_argument("--skip_full_baseline", action="store_true",
                   help="Skip the K^2-frame baseline forward (saves ~1 fwd/sample).")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    K = args.K
    n_frames = K * K
    M = args.M

    # ── Benchmark dispatch ──
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
        logger.info("Benchmark: LongVideoBench")
        if not args.lvb_json:
            raise SystemExit("--benchmark lvb requires --lvb_json")
    else:
        # Lazy V1 import (avoid hard dep on GridProbe.eval.video_mme for V2/LVB users).
        from GridProbe.eval.video_mme import (
            build_prompt as build_prompt_v1,
            check_answer as check_answer_v1,
        )
        LETTERS = ["A", "B", "C", "D"]
        build_prompt = build_prompt_v1
        check_answer = check_answer_v1
        logger.info("Benchmark: Video-MME (4-option A-D)")

    # ── Load QA model ──
    from transformers import AutoProcessor, AutoModelForImageTextToText
    logger.info("Loading QA model %s ...", args.vlm_model)
    kw = dict(torch_dtype=torch.bfloat16)
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    try:
        qa_model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model, attn_implementation="sdpa", **kw)
    except TypeError:
        qa_model = AutoModelForImageTextToText.from_pretrained(args.vlm_model, **kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model = qa_model.to(device).eval()
    proc_kw = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    qa_processor = AutoProcessor.from_pretrained(args.vlm_model, **proc_kw)
    letter_variants = get_letter_token_id_variants(qa_processor.tokenizer, LETTERS)

    qa_dims = infer_lm_dims(qa_model)
    logger.info("QA LM dims for FLOPs accounting: %s", qa_dims)

    # ── Load MDP3 selector ──
    logger.info("Loading MDP3 selector (SigLIP=%s, default M=%d) ...",
                args.siglip_model, M)
    mdp3 = MDP3(device=str(device), n_selection=M, lamda=args.mdp3_lambda,
                siglip_model=args.siglip_model, cache_dir=args.cache_dir,
                return_indices=True)

    # ── Optional per-sample M lookup (matched-compute against GridProbe) ──
    per_sample_M = None
    if args.per_sample_M_from:
        with open(args.per_sample_M_from) as _f:
            _gp = json.load(_f)
        per_sample_M = {}
        for r in _gp.get("results", []) or []:
            rid = r.get("id")
            mu = r.get("M_used")
            if rid is not None and mu is not None:
                per_sample_M[rid] = int(mu)
        logger.info("Per-sample M lookup loaded from %s (%d entries). "
                    "MDP3 n_selection will follow GridProbe's M_eff per "
                    "question; fallback to --M=%d if id missing.",
                    args.per_sample_M_from, len(per_sample_M), M)

    # ── Load samples ──
    if args.benchmark == "video_mme_v2":
        all_samples = load_video_mme_v2(args.data_dir,
                                        with_subtitle=args.with_subtitle)
    elif args.benchmark == "lvb":
        all_samples = load_lvb(args.lvb_json, lvb_dir=args.lvb_dir,
                               with_subtitle=args.with_subtitle)
    else:
        all_samples = load_video_mme(args.data_dir)

    if args.duration_bin:
        all_samples = [s for s in all_samples
                       if s["duration_bin"] == args.duration_bin]
    keep_ids = set()
    if args.filter_ids:
        keep_ids |= {sid.strip() for sid in args.filter_ids.split(",")
                     if sid.strip()}
    if args.filter_ids_file:
        with open(args.filter_ids_file) as _f:
            keep_ids |= {line.strip() for line in _f if line.strip()}
    if keep_ids:
        before = len(all_samples)
        all_samples = [s for s in all_samples if str(s.get("id")) in keep_ids]
        logger.info("Filter: keeping %d / %d samples (from %d ids)",
                    len(all_samples), before, len(keep_ids))
        if not all_samples:
            logger.warning("No samples matched filter; nothing to do.")
    if args.n_shards > 1:
        if not (0 <= args.shard_index < args.n_shards):
            raise ValueError(
                f"--shard_index={args.shard_index} must be in [0, {args.n_shards})")
        before = len(all_samples)
        all_samples = all_samples[args.shard_index::args.n_shards]
        logger.info("Sharding: shard %d/%d → %d / %d",
                    args.shard_index, args.n_shards, len(all_samples), before)
    if args.n_samples and args.n_samples > 0:
        samples = all_samples[:args.n_samples]
    else:
        samples = all_samples
    logger.info("Running %d samples (K=%d, M=%d, MDP3 baseline)",
                len(samples), K, M)

    # ── Counters ──
    METHODS = ["baseline_full", "mdp3"]
    if args.benchmark == "video_mme_v2":
        BINS = BINS_V2 + ["0"]
    elif args.benchmark == "lvb":
        BINS = BINS_LVB
    else:
        BINS = ["short", "medium", "long"]

    correct = {m: 0 for m in METHODS}
    correct_by_bin = {b: {m: 0 for m in METHODS} for b in BINS}
    total = 0
    total_by_bin = {b: 0 for b in BINS}
    timings = {"selector": [], "qa": [], "baseline": []}
    timings_by_bin = {b: {"selector": [], "qa": [], "baseline": []}
                      for b in BINS}
    flops_log = {m: [] for m in METHODS + ["mdp3_selector", "mdp3_qa"]}
    flops_by_bin = {b: {m: [] for m in METHODS + ["mdp3_selector", "mdp3_qa"]}
                    for b in BINS}

    all_results = []

    # ── Resume ──
    done_ids = set()
    if not args.no_resume and Path(args.output).exists():
        try:
            with open(args.output) as _f:
                _existing = json.load(_f)
            for r in _existing.get("results", []) or []:
                rid = r.get("id")
                if rid is None:
                    continue
                bin_name = r.get("duration_bin", "?")
                if bin_name in BINS:
                    total += 1
                    total_by_bin[bin_name] += 1
                    gold = r.get("answer", "")
                    for m in METHODS:
                        pred = r.get(m)
                        if pred is not None and pred == gold:
                            correct[m] += 1
                            correct_by_bin[bin_name][m] += 1
                    for fkey in flops_log:
                        fv = r.get(f"flops_{fkey}")
                        if fv is not None:
                            flops_log[fkey].append(fv)
                            flops_by_bin[bin_name][fkey].append(fv)
                    for tk, jkey in [("selector", "t_mdp3_selector"),
                                      ("qa", "t_mdp3_qa"),
                                      ("baseline", "t_baseline")]:
                        tv = r.get(jkey)
                        if tv is not None:
                            timings[tk].append(tv)
                            timings_by_bin[bin_name][tk].append(tv)
                all_results.append(r)
                done_ids.add(rid)
            if done_ids:
                logger.info("Resumed: %d samples already done", len(done_ids))
        except Exception as e:
            logger.warning("Resume failed: %s", e)
            done_ids = set()

    # ── Atomic save ──
    def _save_state():
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")

        def _avg(lst):
            return float(np.mean(lst)) if lst else 0.0

        with open(tmp, "w") as _f:
            json.dump({
                "benchmark": args.benchmark,
                "method": "mdp3",
                "vlm_model": args.vlm_model,
                "siglip_model": args.siglip_model,
                "K": K, "n_frames": n_frames, "M": M,
                "mdp3_lambda": args.mdp3_lambda,
                "qa_dims": qa_dims,
                "total": total,
                "total_by_bin": total_by_bin,
                "correct": correct,
                "correct_by_bin": correct_by_bin,
                "avg_time_selector": _avg(timings["selector"]),
                "avg_time_qa": _avg(timings["qa"]),
                "avg_time_baseline": _avg(timings["baseline"]),
                "avg_tflops_overall": {
                    "mdp3_selector": _avg(flops_log["mdp3_selector"]) / 1e12,
                    "mdp3_qa": _avg(flops_log["mdp3_qa"]) / 1e12,
                    "mdp3": _avg(flops_log["mdp3"]) / 1e12,
                    "baseline_full": _avg(flops_log["baseline_full"]) / 1e12,
                },
                "results": all_results,
            }, _f, indent=2)
        tmp.replace(out)

    # ── Main loop ──
    for si, sample in enumerate(samples):
        if sample["id"] in done_ids:
            continue
        logger.info("[%d/%d] %s [%s]", si + 1, len(samples),
                    str(sample["id"])[:20], sample.get("duration_bin", "?"))

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
            # ──── Per-sample M (matched-compute mode) ────
            if per_sample_M is not None:
                M_this = per_sample_M.get(sample["id"], M)
                if sample["id"] not in per_sample_M:
                    logger.debug("  id=%s not in M-lookup; falling back to M=%d",
                                 sample["id"], M)
                # Clamp: MDP3 needs n_selection >= 1 and <= K^2.
                M_this = max(1, min(M_this, n_frames))
                mdp3.set_n_selection(M_this)
            else:
                M_this = M

            # ──── MDP3 selection (SigLIP + DPP) ────
            t0 = time.perf_counter()
            selected_indices = mdp3(frames, prompt_text)
            t_selector = time.perf_counter() - t0
            timings["selector"].append(t_selector)

            # ──── QA forward on selected frames ────
            t0 = time.perf_counter()
            mdp3_out = mdp3_focused(
                qa_model, qa_processor, prompt_text, frames,
                selected_indices, letter_variants, device,
                args.max_video_pixels_focus)
            t_qa = time.perf_counter() - t0
            timings["qa"].append(t_qa)

            # ──── Baseline full pass (K^2 frames) ────
            if args.skip_full_baseline:
                base_full = {"pred": "(skip)", "lp": None}
                t_base = 0.0
            else:
                t0 = time.perf_counter()
                base_full = run_baseline_full(
                    qa_model, qa_processor, prompt_text, frames,
                    letter_variants, device, args.max_video_pixels_focus)
                t_base = time.perf_counter() - t0
                timings["baseline"].append(t_base)

            # ──── Token counts + analytical FLOPs ────
            try:
                _peek_full = build_inputs(qa_processor, prompt_text, frames,
                                          args.max_video_pixels_focus)
                n_tok_full = int(_peek_full["input_ids"].shape[1])
            except Exception:
                n_tok_full = -1
            try:
                qa_frames = [frames[i] for i in mdp3_out["frame_indices"]]
                _peek_qa = build_inputs(qa_processor, prompt_text, qa_frames,
                                        args.max_video_pixels_focus)
                n_tok_qa = int(_peek_qa["input_ids"].shape[1])
            except Exception:
                n_tok_qa = -1

            f_full = lm_flops_per_forward(n_tok_full, qa_dims)
            f_mdp3_qa = lm_flops_per_forward(n_tok_qa, qa_dims)
            # Selector cost: SigLIP image+text passes (analytical estimate).
            # n_text_chunks depends on prompt length post-tokenization. Default
            # to a safe over-estimate of 4 (paper uses up to 64-token chunks).
            f_mdp3_sel = estimate_mdp3_selector_flops(n_frames, n_text_chunks=4)
            f_mdp3_total = f_mdp3_sel + f_mdp3_qa

            # ──── Tally ────
            c_base = check_answer(base_full["pred"], answer, options)
            c_mdp3 = check_answer(mdp3_out["pred"], answer, options)

            bin_name = sample.get("duration_bin", "?")
            if bin_name not in BINS:
                bin_name = None

            correct["baseline_full"] += c_base
            correct["mdp3"] += c_mdp3
            flops_log["baseline_full"].append(f_full)
            flops_log["mdp3_selector"].append(f_mdp3_sel)
            flops_log["mdp3_qa"].append(f_mdp3_qa)
            flops_log["mdp3"].append(f_mdp3_total)
            total += 1
            if bin_name is not None:
                correct_by_bin[bin_name]["baseline_full"] += c_base
                correct_by_bin[bin_name]["mdp3"] += c_mdp3
                flops_by_bin[bin_name]["baseline_full"].append(f_full)
                flops_by_bin[bin_name]["mdp3_selector"].append(f_mdp3_sel)
                flops_by_bin[bin_name]["mdp3_qa"].append(f_mdp3_qa)
                flops_by_bin[bin_name]["mdp3"].append(f_mdp3_total)
                total_by_bin[bin_name] += 1
                timings_by_bin[bin_name]["selector"].append(t_selector)
                timings_by_bin[bin_name]["qa"].append(t_qa)
                if t_base > 0:
                    timings_by_bin[bin_name]["baseline"].append(t_base)

            logger.info(
                "  [%s] ans=%s M=%d | mdp3=%s(%s,%.1fT total) "
                "base=%s(%s,%.1fT) | t_sel=%.1fs t_qa=%.1fs t_base=%.1fs",
                bin_name or "?", answer, M_this,
                mdp3_out["pred"], "OK" if c_mdp3 else "XX", f_mdp3_total / 1e12,
                base_full["pred"], "OK" if c_base else "XX", f_full / 1e12,
                t_selector, t_qa, t_base)

            all_results.append({
                "id": sample["id"],
                "duration_bin": sample.get("duration_bin", "?"),
                "question": sample["question"][:80],
                "answer": answer,
                "M_used": M_this,  # per-sample M actually used by MDP3 this run
                "mdp3": mdp3_out["pred"],
                "baseline_full": base_full["pred"],
                "mdp3_selected_frames": mdp3_out["frame_indices"],
                "t_mdp3_selector": t_selector,
                "t_mdp3_qa": t_qa,
                "t_mdp3_total": t_selector + t_qa,
                "t_baseline": t_base,
                "n_tok_baseline_full": n_tok_full,
                "n_tok_mdp3_qa": n_tok_qa,
                "flops_baseline_full": f_full,
                "flops_mdp3_selector": f_mdp3_sel,
                "flops_mdp3_qa": f_mdp3_qa,
                "flops_mdp3": f_mdp3_total,
            })

        except Exception as e:
            logger.warning("  Failed: %s", e)
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
        finally:
            try:
                del frames
            except Exception:
                pass
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.checkpoint_every > 0 and (si + 1) % args.checkpoint_every == 0:
            try:
                _save_state()
                logger.info("  [checkpoint] saved %d/%d to %s",
                            si + 1, len(samples), args.output)
            except Exception as _e:
                logger.warning("  [checkpoint failed] %s", _e)

    # ── Summary ──
    print("\n" + "=" * 92)
    print(f"  MDP3 BASELINE EVAL (benchmark={args.benchmark}, K={K}, M={M}, "
          f"n={total})")
    print("=" * 92)

    def _avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    def _block(n, c_block, t_block, f_block, header):
        if n <= 0:
            print(f"  ({header}: no samples)")
            return
        print(f"  ── {header}  (n={n}) ──")
        print(f"  {'Method':<28} {'Acc':>7} {'Correct':>10} "
              f"{'Avg time':>10} {'Avg TFLOPs':>11}")
        print("  " + "-" * 75)
        for m in METHODS:
            acc = c_block[m] / n * 100 if n else 0.0
            if m == "mdp3":
                avg_t = _avg(t_block["selector"]) + _avg(t_block["qa"])
                t_str = f"{avg_t:.2f}s (sel+qa)"
            else:
                avg_t = _avg(t_block["baseline"])
                t_str = f"{avg_t:.2f}s"
            avg_f = _avg(f_block.get(m, []))
            label = f"MDP3 (M={M})" if m == "mdp3" else f"baseline (K^2={n_frames})"
            print(f"  {label:<28} {acc:>6.1f}%  {c_block[m]:>4}/{n:<4} "
                  f"{t_str:>16} {avg_f / 1e12:>10.2f}T")
        # Selector vs QA breakdown for MDP3.
        sel_t = _avg(t_block["selector"])
        qa_t = _avg(t_block["qa"])
        sel_f = _avg(f_block.get("mdp3_selector", []))
        qa_f = _avg(f_block.get("mdp3_qa", []))
        if sel_t > 0 or qa_t > 0:
            print(f"      [breakdown] selector={sel_t:.2f}s "
                  f"({sel_f / 1e12:.2f}T) | qa={qa_t:.2f}s "
                  f"({qa_f / 1e12:.2f}T)")
        print()

    for bn in BINS:
        header_label = bn
        if args.benchmark == "video_mme_v2":
            header_label = (f"LEVEL {bn}" if bn in ("1", "2", "3")
                            else "UNLEVELED (group q1-q3)")
        else:
            header_label = bn.upper()
        _block(total_by_bin[bn], correct_by_bin[bn],
               timings_by_bin[bn], flops_by_bin[bn], header_label)
    _block(total, correct, timings, flops_log, "OVERALL")

    if total > 0:
        mdp3_acc = correct["mdp3"] / total * 100
        base_acc = correct["baseline_full"] / total * 100
        print(f"  Δ MDP3 vs baseline (K^2={n_frames}):  "
              f"{mdp3_acc - base_acc:+.1f}% (overall)")
        if flops_log["mdp3"] and flops_log["baseline_full"]:
            avg_mdp3 = _avg(flops_log["mdp3"])
            avg_base = _avg(flops_log["baseline_full"])
            print(f"  TFLOPs ratio (mdp3 / baseline_full):  "
                  f"{avg_mdp3 / avg_base:.2f}×")
    print("=" * 92)

    _save_state()
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
