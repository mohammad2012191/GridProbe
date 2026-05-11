"""
Convert mdp3_eval.py output to two_stage_eval.py schema for analyze_results.py.
==============================================================================
analyze_results.py is hardwired to the field names of two_stage_eval.py
(`two_stage`, `flops_two_stage`, `t_stage1`, ...). MDP3's output uses
`mdp3`, `flops_mdp3`, `t_mdp3_*`. This script renames keys so that
analyze_results.py reads MDP3's results in the slot it expects "GridProbe".

Per-sample remapping:
    mdp3                    -> two_stage             (the prediction)
    mdp3_selected_frames    -> golden_frames
    t_mdp3_selector         -> t_stage1              (selector wall time)
    t_mdp3_qa               -> t_stage2              (QA wall time)
    n_tok_mdp3_qa           -> n_tok_two_stage
    flops_mdp3_selector     -> flops_probe_ensemble  (selector compute)
    flops_mdp3_qa           -> flops_two_stage_qa_only  (informational)
    flops_mdp3              -> flops_two_stage       (total compute)

Top-level remapping mirrors the same renames in `correct`, `correct_by_bin`,
`avg_tflops_overall`, etc.

Optional: --merge_baseline_from <gridprobe_json> pulls full-pool baseline
predictions and FLOPs from a parallel GridProbe JSON (same samples, different
method) so the converted file has real baseline numbers instead of (skip).

Usage:
    python -m GridProbe.eval.mdp3_to_analyze_format \
        --input results/mdp3_v2_2B_K12_M8_merged.json \
        --output results/mdp3_v2_2B_K12_M8_for_analyze.json \
        --merge_baseline_from results/two_stage_v2_2B_K12_auto.json

    python -m GridProbe.eval.analyze_results \
        --input results/mdp3_v2_2B_K12_M8_for_analyze.json \
        --parquet /path/to/test.parquet \
        --out_dir analysis_mdp3_v2/
"""

import argparse
import json
from pathlib import Path


PER_SAMPLE_RENAMES = {
    "mdp3":                  "two_stage",
    "mdp3_selected_frames":  "golden_frames",
    "t_mdp3_selector":       "t_stage1",
    "t_mdp3_qa":             "t_stage2",
    "n_tok_mdp3_qa":         "n_tok_two_stage",
    "flops_mdp3_selector":   "flops_probe_ensemble",
    "flops_mdp3":            "flops_two_stage",
    # leave flops_mdp3_qa as-is (informational; analyzer doesn't use it)
}


def remap_per_sample(row):
    """Apply field renames to one results-row dict."""
    out = dict(row)
    for src, dst in PER_SAMPLE_RENAMES.items():
        if src in out:
            out[dst] = out.pop(src)
    return out


def remap_top_level(blob, has_real_baseline=False):
    """Remap the summary dicts that analyze_results.py also reads."""
    out = dict(blob)

    # correct: { "mdp3": N } -> { "two_stage": N }
    if "correct" in out and isinstance(out["correct"], dict):
        c = dict(out["correct"])
        if "mdp3" in c:
            c["two_stage"] = c.pop("mdp3")
        out["correct"] = c

    if "correct_by_bin" in out and isinstance(out["correct_by_bin"], dict):
        cb = {}
        for bn, mdict in out["correct_by_bin"].items():
            md = dict(mdict)
            if "mdp3" in md:
                md["two_stage"] = md.pop("mdp3")
            cb[bn] = md
        out["correct_by_bin"] = cb

    # avg_tflops_overall: { "mdp3": x, "mdp3_selector": y, "mdp3_qa": z }
    #   -> { "two_stage": x, "probe_ensemble": y, ... }
    if "avg_tflops_overall" in out and isinstance(out["avg_tflops_overall"], dict):
        f = dict(out["avg_tflops_overall"])
        if "mdp3" in f:
            f["two_stage"] = f.pop("mdp3")
        if "mdp3_selector" in f:
            f["probe_ensemble"] = f.pop("mdp3_selector")
        # drop mdp3_qa (no analyzer slot for it)
        f.pop("mdp3_qa", None)
        out["avg_tflops_overall"] = f

    # Schema marker so a downstream tool can tell this is converted.
    out["_converted_from"] = "mdp3_eval.py"
    return out


def merge_baseline_from(blob, baseline_path):
    """Replace baseline_full predictions / FLOPs / timings with values from
    a parallel two_stage_eval.py JSON keyed by sample id.

    Useful when the MDP3 run used --skip_full_baseline and we still want
    to show baseline numbers in the analyzer output.
    """
    with open(baseline_path) as f:
        gp = json.load(f)
    gp_by_id = {r["id"]: r for r in gp.get("results", []) or []}

    n_patched = 0
    n_missing = 0
    for r in blob.get("results", []):
        rid = r.get("id")
        gpr = gp_by_id.get(rid)
        if not gpr:
            n_missing += 1
            continue
        # Only pull baseline-related fields; leave method-specific fields alone.
        for k in ("baseline_full", "baseline_uniform_M", "baseline_random_M",
                  "t_baseline", "t_uniform", "t_random",
                  "n_tok_baseline_full", "n_tok_uniform_M", "n_tok_random_M",
                  "flops_baseline_full", "flops_baseline_uniform_M",
                  "flops_baseline_random_M",
                  "row_confs", "col_confs"):  # row/col confs help the V2 stats
            if k in gpr:
                r[k] = gpr[k]
        n_patched += 1

    # Re-aggregate baseline counters from per-sample data.
    correct_full = 0
    correct_uni = 0
    correct_random = 0
    correct_full_by_bin = {}
    correct_uni_by_bin = {}
    correct_random_by_bin = {}

    for r in blob.get("results", []):
        gold = r.get("answer")
        bn = r.get("duration_bin")
        for tag, accum, accum_bin in [
            ("baseline_full", "f", correct_full_by_bin),
            ("baseline_uniform_M", "u", correct_uni_by_bin),
            ("baseline_random_M", "r", correct_random_by_bin),
        ]:
            pred = r.get(tag)
            if pred and pred != "(skip)" and pred == gold:
                if tag == "baseline_full":
                    correct_full += 1
                elif tag == "baseline_uniform_M":
                    correct_uni += 1
                else:
                    correct_random += 1
                accum_bin[bn] = accum_bin.get(bn, 0) + 1

    # Patch summary blocks.
    c = blob.setdefault("correct", {})
    c["baseline_full"] = correct_full
    c["baseline_uniform_M"] = correct_uni
    c["baseline_random_M"] = correct_random
    cb = blob.setdefault("correct_by_bin", {})
    for bn, mdict in cb.items():
        mdict["baseline_full"] = correct_full_by_bin.get(bn, 0)
        mdict["baseline_uniform_M"] = correct_uni_by_bin.get(bn, 0)
        mdict["baseline_random_M"] = correct_random_by_bin.get(bn, 0)

    print(f"[merge_baseline] patched {n_patched} rows from {baseline_path}")
    if n_missing:
        print(f"[merge_baseline] WARN: {n_missing} samples not found in "
              f"baseline JSON (kept (skip) values)")
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="MDP3 eval JSON (e.g., merged shards).")
    ap.add_argument("--output", required=True,
                    help="Path to write the converted (two_stage-formatted) JSON.")
    ap.add_argument("--merge_baseline_from", default=None,
                    help="Optional parallel two_stage_eval.py JSON to pull "
                         "real baseline_full/_uniform/_random preds from "
                         "(MDP3 runs typically use --skip_full_baseline).")
    args = ap.parse_args()

    with open(args.input) as f:
        blob = json.load(f)

    # Per-sample remap
    blob["results"] = [remap_per_sample(r) for r in blob.get("results", [])]
    # Top-level remap
    blob = remap_top_level(blob)
    # Optional: pull baseline numbers from a parallel GridProbe JSON
    if args.merge_baseline_from:
        blob = merge_baseline_from(blob, args.merge_baseline_from)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2, default=str)

    n = len(blob.get("results", []))
    print(f"[convert] wrote {n} samples to {out_path}")
    print(f"[convert] you can now run:")
    print(f"  python -m GridProbe.eval.analyze_results \\")
    print(f"      --input {out_path} \\")
    print(f"      --parquet /path/to/test.parquet \\")
    print(f"      --out_dir analysis_mdp3/")


if __name__ == "__main__":
    main()
