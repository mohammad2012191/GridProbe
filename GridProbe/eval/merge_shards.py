"""
Merge per-shard JSON outputs from `two_stage_eval.py` or
`two_stage_eval_crossmodel.py` into one aggregated result file.

After running 8 sharded jobs:
    for i in 0..7: --n_shards 8 --shard_index $i --output ts_shard${i}.json

Run:
    python -m GridProbe.eval.merge_shards \
        --inputs ts_shard0.json ts_shard1.json ... ts_shard7.json \
        --output ts_merged.json

Outputs:
  - Concatenated per-sample results (de-duplicated by question id if a question
    accidentally appears in multiple shards)
  - Recomputed correct counts (overall + per bin)
  - Recomputed avg time and avg TFLOPs (overall + per bin)
  - Header sanity-checks: K, benchmark, model(s), pr_gamma must match.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# Methods we know about. A shard's `correct` dict tells us which set is in use.
ALL_METHODS = [
    "baseline_full", "baseline_uniform_M", "baseline_random_M",
    "probe_ensemble", "two_stage",
    # cross-model methods
    "qa_full", "qa_native", "qa_uniform_M", "probe_only",
    # mdp3_eval.py methods
    "mdp3",
]


def load_shards(paths):
    shards = []
    for p in paths:
        with open(p) as f:
            shards.append(json.load(f))
    return shards


def must_match(shards, key):
    """Sanity-check that all shards share a config field."""
    vals = [s.get(key) for s in shards]
    distinct = {json.dumps(v, sort_keys=True, default=str) for v in vals}
    if len(distinct) > 1:
        sys.exit(f"Shards disagree on '{key}': {distinct}")
    return vals[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Paths to shard JSON files (or glob pattern with quotes).")
    p.add_argument("--output", default="ts_merged.json")
    p.add_argument("--allow_mismatch", action="store_true",
                   help="Skip the sanity-check on K / benchmark / model(s).")
    args = p.parse_args()

    # Expand globs
    paths = []
    for pat in args.inputs:
        if any(c in pat for c in "*?[]"):
            paths.extend(sorted(glob.glob(pat)))
        else:
            paths.append(pat)
    paths = sorted(set(paths))
    if not paths:
        sys.exit("No input files found")
    print(f"→ Merging {len(paths)} shards:")
    for p_ in paths:
        print(f"  - {p_}")

    shards = load_shards(paths)

    # ── Sanity checks ──
    if not args.allow_mismatch:
        for key in ["K", "benchmark", "selector_model", "qa_model",
                    "selector", "M_mode", "pr_gamma", "model_dims", "qa_dims",
                    "selector_dims"]:
            # Some keys are only in cross-model output, others only in single-model.
            if any(key in s for s in shards):
                must_match(shards, key)

    # Detect which method set is in use (single-model vs cross-model)
    sample_correct = next((s for s in shards if "correct" in s), {}).get("correct", {})
    methods_in_use = [m for m in ALL_METHODS if m in sample_correct]
    if not methods_in_use:
        sys.exit("Could not identify methods: shard JSONs missing 'correct' block.")

    # ── Bin labels ──
    # Try to infer from the union of by_bin keys
    bin_keys = set()
    for s in shards:
        bin_keys.update(s.get("correct_by_bin", {}).keys())
        bin_keys.update(s.get("total_by_bin", {}).keys())
    bins = sorted(bin_keys)

    # ── Aggregate ──
    total = 0
    total_by_bin = {b: 0 for b in bins}
    correct = {m: 0 for m in methods_in_use}
    correct_by_bin = {b: {m: 0 for m in methods_in_use} for b in bins}

    # We aggregate AVG times/flops by re-deriving from per-sample data when present;
    # otherwise we fall back to weighted average across shards.
    # Per-sample data is in shard["results"] — list of dicts.
    seen_ids = set()
    merged_results = []
    flops_lists = {m: [] for m in methods_in_use}
    flops_lists_by_bin = {b: {m: [] for m in methods_in_use} for b in bins}
    time_lists = {}
    time_lists_by_bin = {b: {} for b in bins}

    # Method → JSON field name(s) for per-sample time and flops.
    # Single-model uses keys like t_baseline / flops_baseline_full;
    # cross-model uses t_qa_full / flops_qa_full.
    TIME_FIELD = {
        "baseline_full":      "t_baseline",
        "baseline_uniform_M": "t_uniform",
        "baseline_random_M":  "t_random",
        "probe_ensemble":     "t_stage1",
        "two_stage":          "t_stage2",     # focused only; total = stage1+stage2
        "qa_full":            "t_qa_full",
        "qa_native":          "t_qa_native",
        "qa_uniform_M":       "t_qa_uniform_M",
        "probe_only":         "t_probe",
        "mdp3":               "t_mdp3_total",
    }
    FLOPS_FIELD = {
        "baseline_full":      "flops_baseline_full",
        "baseline_uniform_M": "flops_baseline_uniform_M",
        "baseline_random_M":  "flops_baseline_random_M",
        "probe_ensemble":     "flops_probe_ensemble",
        "two_stage":          "flops_two_stage",
        "qa_full":            "flops_qa_full",
        "qa_native":          "flops_qa_native",
        "qa_uniform_M":       "flops_qa_uniform_M",
        "probe_only":         "flops_probe",
        "mdp3":               "flops_mdp3",
    }

    adaptive_Ms = []

    for s in shards:
        for r in s.get("results", []):
            rid = r.get("id") or r.get("question_id") or r.get("q_id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            merged_results.append(r)

            bn = r.get("duration_bin")
            total += 1
            if bn in bins:
                total_by_bin[bn] += 1

            # Per-method correct (re-derive from prediction == answer using shard's check)
            # We don't re-run check_answer here; instead trust the shard's own correctness
            # via the shard's `correct` dict — but `results` rows don't include c_* booleans
            # by default in single-model JSON.
            # Fallback: trust shard `correct` block; do simple weighted aggregation below.
            # BUT we still want per-sample FLOPs/timings for accurate avgs.
            for m in methods_in_use:
                tf = TIME_FIELD.get(m)
                if tf and tf in r:
                    time_lists.setdefault(m, []).append(float(r[tf]))
                    if bn in bins:
                        time_lists_by_bin[bn].setdefault(m, []).append(float(r[tf]))
                ff = FLOPS_FIELD.get(m)
                if ff and ff in r and r[ff]:
                    flops_lists[m].append(float(r[ff]))
                    if bn in bins:
                        flops_lists_by_bin[bn][m].append(float(r[ff]))

            if "M_used" in r:
                try:
                    adaptive_Ms.append(int(r["M_used"]))
                except Exception:
                    pass

    # Per-method correct: re-aggregate by trusting shard counters (cheap and exact)
    for s in shards:
        c_block = s.get("correct", {})
        for m in methods_in_use:
            correct[m] += int(c_block.get(m, 0))
        cb = s.get("correct_by_bin", {})
        for b in bins:
            for m in methods_in_use:
                correct_by_bin[b][m] += int(cb.get(b, {}).get(m, 0))

    # ── Merge config ──
    base_cfg = {}
    for s in shards[0:1]:
        for k, v in s.items():
            if k in {"results", "correct", "correct_by_bin", "total", "total_by_bin",
                     "avg_time_overall", "avg_time_by_bin",
                     "avg_tflops_overall", "avg_tflops_by_bin",
                     "adaptive_Ms"}:
                continue
            base_cfg[k] = v

    # ── Compute averaged metrics ──
    def mean_or_zero(xs):
        return float(np.mean(xs)) if xs else 0.0

    avg_time_overall = {m: mean_or_zero(time_lists.get(m, [])) for m in methods_in_use}
    avg_tflops_overall = {m: mean_or_zero(flops_lists.get(m, [])) / 1e12
                          for m in methods_in_use}
    avg_time_by_bin = {
        b: {m: mean_or_zero(time_lists_by_bin[b].get(m, [])) for m in methods_in_use}
        for b in bins
    }
    avg_tflops_by_bin = {
        b: {m: mean_or_zero(flops_lists_by_bin[b].get(m, [])) / 1e12
            for m in methods_in_use}
        for b in bins
    }

    merged = {
        **base_cfg,
        "n_shards_merged": len(shards),
        "shard_paths": [str(Path(p_).resolve()) for p_ in paths],
        "total":          total,
        "total_by_bin":   total_by_bin,
        "correct":        correct,
        "correct_by_bin": correct_by_bin,
        "avg_time_overall":   avg_time_overall,
        "avg_time_by_bin":    avg_time_by_bin,
        "avg_tflops_overall": avg_tflops_overall,
        "avg_tflops_by_bin":  avg_tflops_by_bin,
        "adaptive_Ms": adaptive_Ms,
        "results":     merged_results,
    }

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    # ── Print summary ──
    print()
    print(f"✓ Merged {total} unique samples into {args.output}")
    print(f"  Bins: {bins}")
    for b in bins:
        print(f"    {b}: {total_by_bin[b]} samples")
    print()
    print(f"  {'Method':<28} {'Acc':>7} {'Correct':>10} {'Avg time':>10} {'Avg TFLOPs':>11}")
    print("  " + "-" * 72)
    for m in methods_in_use:
        acc = correct[m] / total * 100 if total else 0.0
        print(f"  {m:<28} {acc:>6.1f}%  {correct[m]:>4}/{total:<4} "
              f"{avg_time_overall[m]:>9.2f}s {avg_tflops_overall[m]:>10.2f}T")


if __name__ == "__main__":
    main()
