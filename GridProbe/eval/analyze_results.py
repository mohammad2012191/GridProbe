"""
Post-process a single eval JSON into the plots and tables the paper needs.

Reads the per-sample `results[]` array from a `two_stage_eval.py` output and
produces:

  1. tflops_dist.png        — Histogram of TFLOPs per question, baseline vs
                              two_stage, with kurtosis annotation. (Figure 6a)
  2. M_dist.png             — Adaptive M histogram, optionally split by level.
  3. tflops_vs_correct.png  — Per-sample scatter (TFLOPs on x, correct 0/1 on y),
                              jittered, two_stage points only.
  4. method_x_level.tex     — LaTeX-style accuracy table by method × level.
  5. winloss.tex            — For each level, fraction of (2stage_wins / both_wrong /
                              baseline_wins / both_correct).
  6. per_task_type.tex      — (optional) Per-task-type breakdown, requires the
                              V2 parquet to look up `third_head` for each question.
  7. nonlin_v2.tex          — (optional) V2 grouped non-linear score, requires
                              the parquet for group_type and group_structure.

Usage:
    python -m GridProbe.eval.analyze_results \
        --input  ts_v2_2b_k8_skew.json \
        --parquet /path/to/Video-MME-v2/test.parquet \
        --out_dir analysis_k8 \
        --label "K=8 skew"

Without --parquet, items 6 and 7 are skipped.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_parquet_with_meta(parquet_path):
    """Load V2 parquet, return dict {question_id: row_dict}."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    out = {}
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        out[qid] = {
            "video_id":        str(row.get("video_id", "")),
            "group_type":      str(row.get("group_type", "")),
            "group_structure": str(row.get("group_structure", "")),
            "level":           str(row.get("level") if row.get("level") is not None else ""),
            "second_head":     str(row.get("second_head") or ""),
            "third_head":      str(row.get("third_head") or ""),
        }
    return out


# ─────────────────────────────────────────────────────────────
# Per-sample correctness extraction
# ─────────────────────────────────────────────────────────────

def per_sample_correct(samples):
    """Compute per-method correctness flags using the lenient V2 checker."""
    from GridProbe.eval.video_mme_v2 import check_answer_v2 as check
    methods = ["baseline_full", "baseline_uniform_M", "baseline_random_M",
               "probe_ensemble", "two_stage"]
    per = []
    for r in samples:
        ans = r.get("answer", "")
        out = {"id": r.get("id", ""), "level": r.get("duration_bin", ""), "M": r.get("M_used", 0)}
        for m in methods:
            pred = r.get(m, "")
            out[m + "_correct"] = check(pred, ans, None)
        out["flops_baseline"]  = float(r.get("flops_baseline_full", 0))
        out["flops_two_stage"] = float(r.get("flops_two_stage", 0))
        out["flops_probe"]     = float(r.get("flops_probe_ensemble", 0))
        per.append(out)
    return per, methods


# ─────────────────────────────────────────────────────────────
# Plotting (matplotlib, with one PIL fallback for the M histogram)
# ─────────────────────────────────────────────────────────────

def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable ({e}); skipping plots that require it.")
        return None


def shape_stats(arr):
    """Compute the three shape statistics we report.

    Matches the distribution-shape statistic used by the selector:
        σ = |skew| + 0.5 · max(0, kurt_excess)

    Standardized moments are scale-invariant, so the same σ used on the
    importance map per question is meaningful on the cross-question TFLOPs
    distribution: high σ → distribution is peaked / asymmetric (i.e., the
    method allocates compute non-uniformly across questions).

    Returns:
        dict with: skew (signed), kurt_excess, sigma, std, cv
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = {"skew": 0.0, "kurt_excess": 0.0, "sigma": 0.0,
           "std": 0.0, "cv": 0.0, "mean": 0.0}
    if len(arr) < 2:
        return out
    mu = float(arr.mean())
    sd = float(arr.std())
    out["mean"] = mu
    out["std"]  = sd
    out["cv"]   = sd / mu if mu else 0.0
    if sd < 1e-10:
        return out
    skew = float(((arr - mu) ** 3).mean() / (sd ** 3))
    kurt_ex = float(((arr - mu) ** 4).mean() / (sd ** 4) - 3.0)
    out["skew"] = skew
    out["kurt_excess"] = kurt_ex
    out["sigma"] = abs(skew) + 0.5 * max(0.0, kurt_ex)
    return out


def plot_tflops_distribution(per, out_path, label=""):
    """Two-panel ridgeline-style histogram: baseline (top) vs GridProbe (bottom)
    on the same x-axis, with mean lines, ±1σ shading, and a compact stats box.

    Designed for paper readability — the baseline's near-delta spike and
    GridProbe's broad distribution are both legible at a glance.
    """
    plt = _try_matplotlib()
    if plt is None:
        return
    base = np.array([p["flops_baseline"]  for p in per]) / 1e12
    twos = np.array([p["flops_two_stage"] for p in per]) / 1e12
    s_b = shape_stats(base)
    s_t = shape_stats(twos)

    # Shared x-range with small headroom
    x_lo = 0
    x_hi = max(base.max(), twos.max()) * 1.05
    bins = np.linspace(x_lo, x_hi, 80)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.10})

    # ---- Baseline panel (top) ----
    base_color = "#7a7a7a"
    ax_top.hist(base, bins=bins, color=base_color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
    ax_top.axvline(s_b["mean"], color=base_color, linestyle="--", linewidth=1.4,
                   label=f"mean = {s_b['mean']:.1f}T")
    # ±1σ band (so narrow it's basically a line; that IS the point)
    ax_top.axvspan(s_b["mean"] - s_b["std"], s_b["mean"] + s_b["std"],
                   color=base_color, alpha=0.18)
    ax_top.set_ylabel("# questions", fontsize=10)
    ax_top.set_title("Baseline (no selection): near-constant compute",
                     fontsize=11, color=base_color, loc="left", pad=4)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    # Stats box for baseline
    ax_top.text(0.985, 0.93,
        f"mean={s_b['mean']:.1f}T   std={s_b['std']:.2f}T   "
        f"CV={s_b['cv']:.3f}   σ={s_b['sigma']:.2f}",
        transform=ax_top.transAxes, fontsize=9, ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=base_color, lw=0.8))

    # ---- GridProbe panel (bottom) ----
    ours_color = "#1f6fb4"
    ax_bot.hist(twos, bins=bins, color=ours_color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
    ax_bot.axvline(s_t["mean"], color=ours_color, linestyle="--", linewidth=1.4,
                   label=f"mean = {s_t['mean']:.1f}T")
    ax_bot.axvspan(s_t["mean"] - s_t["std"], s_t["mean"] + s_t["std"],
                   color=ours_color, alpha=0.15)
    # Also draw baseline mean line on bottom panel as a reference
    ax_bot.axvline(s_b["mean"], color=base_color, linestyle=":", linewidth=1.0,
                   alpha=0.7)
    ax_bot.set_xlabel("TFLOPs per question", fontsize=10)
    ax_bot.set_ylabel("# questions", fontsize=10)
    ax_bot.set_title("GridProbe (ours): adaptive, broadly distributed",
                     fontsize=11, color=ours_color, loc="left", pad=4)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    # Stats box for GridProbe
    ax_bot.text(0.985, 0.93,
        f"mean={s_t['mean']:.1f}T   std={s_t['std']:.2f}T   "
        f"CV={s_t['cv']:.3f}   σ={s_t['sigma']:.2f}",
        transform=ax_bot.transAxes, fontsize=9, ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ours_color, lw=0.8))

    # Headline ratio annotation between panels
    cv_ratio = (s_t["cv"] / s_b["cv"]) if s_b["cv"] > 1e-9 else float("inf")
    cost_ratio = (s_t["mean"] / s_b["mean"]) if s_b["mean"] > 1e-9 else float("inf")
    fig.suptitle(
        rf"Per-question compute: $\bf{{{cost_ratio:.2f}\times}}$ avg cost, "
        rf"$\bf{{{cv_ratio:.0f}\times}}$ more variable across questions"
        + (f"  ({label})" if label else ""),
        fontsize=12, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  → {out_path}")
    print(f"     Baseline:  mean={s_b['mean']:.2f}T  std={s_b['std']:.2f}T  "
          f"CV={s_b['cv']:.4f}  σ={s_b['sigma']:.3f}  "
          f"skew={s_b['skew']:+.3f}  kurt_ex={s_b['kurt_excess']:.3f}")
    print(f"     GridProbe: mean={s_t['mean']:.2f}T  std={s_t['std']:.2f}T  "
          f"CV={s_t['cv']:.4f}  σ={s_t['sigma']:.3f}  "
          f"skew={s_t['skew']:+.3f}  kurt_ex={s_t['kurt_excess']:.3f}")
    print(f"     Headline: {cost_ratio:.2f}× cost, {cv_ratio:.0f}× CV ratio")


def plot_M_dist(per, out_path, label=""):
    plt = _try_matplotlib()
    if plt is None:
        return
    Ms = np.array([p["M"] for p in per])
    levels = sorted({p["level"] for p in per if p["level"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(int(Ms.min()), int(Ms.max()) + 2) - 0.5
    if len(levels) > 1:
        for lv in levels:
            sub = Ms[[p["level"] == lv for p in per]]
            ax.hist(sub, bins=bins, alpha=0.55, label=f"Level {lv} (n={len(sub)})", edgecolor="white")
    else:
        ax.hist(Ms, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel(r"Adaptive $M_{\rm eff}$")
    ax.set_ylabel("# questions")
    ax.set_title(f"Distribution of selected frames{(' — ' + label) if label else ''}")
    if len(levels) > 1:
        ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}  (M: min={Ms.min()} median={int(np.median(Ms))} mean={Ms.mean():.1f} max={Ms.max()})")


def plot_tflops_vs_correct(per, out_path, label=""):
    plt = _try_matplotlib()
    if plt is None:
        return
    twos = np.array([p["flops_two_stage"] for p in per]) / 1e12
    correct = np.array([1 if p["two_stage_correct"] else 0 for p in per], dtype=np.float64)
    # jitter
    rng = np.random.default_rng(42)
    correct_j = correct + rng.uniform(-0.06, 0.06, size=len(correct))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(twos[correct == 0], correct_j[correct == 0], s=6, alpha=0.45, color="#cc4444", label="incorrect")
    ax.scatter(twos[correct == 1], correct_j[correct == 1], s=6, alpha=0.45, color="#2c8a2c", label="correct")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["incorrect", "correct"])
    ax.set_xlabel("Two-stage TFLOPs per question")
    ax.set_title(f"TFLOPs vs correctness{(' — ' + label) if label else ''}")
    ax.legend(frameon=False, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────

def table_method_x_level(per, methods, out_path):
    levels = sorted({p["level"] for p in per if p["level"]}) or ["?"]
    by = {(m, lv): [0, 0] for m in methods for lv in levels + ["overall"]}  # [correct, total]
    for p in per:
        lv = p["level"] or "?"
        for m in methods:
            by[(m, lv)][0] += int(p[m + "_correct"])
            by[(m, lv)][1] += 1
            by[(m, "overall")][0] += int(p[m + "_correct"])
            by[(m, "overall")][1] += 1

    pretty = {"baseline_full": "Baseline ($K^2$)",
              "baseline_uniform_M": "Uniform-$K^2$",
              "baseline_random_M": "Random-$K^2$",
              "probe_ensemble":    "Probe ensemble",
              "two_stage":         "GridProbe (ours)"}
    cols = levels + ["overall"]

    lines = []
    lines.append(r"% Auto-generated by analyze_results.py")
    lines.append(r"\begin{tabular}{l " + "r " * len(cols) + r"}")
    lines.append(r"\toprule")
    lines.append("Method & " + " & ".join(f"L{c}" if c != "overall" else "Overall" for c in cols) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        cells = []
        for lv in cols:
            corr, tot = by[(m, lv)]
            cells.append(f"{(corr/tot*100 if tot else 0):.1f}")
        lines.append(pretty.get(m, m) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → {out_path}")


def table_winloss(per, out_path):
    """For each level, count: 2stage wins (2s right, base wrong) / both right /
    both wrong / baseline wins (base right, 2s wrong)."""
    levels = sorted({p["level"] for p in per if p["level"]})
    cats = ["both_right", "2s_wins", "base_wins", "both_wrong"]

    table = {(lv, c): 0 for lv in levels + ["overall"] for c in cats}
    for p in per:
        b, t = bool(p["baseline_full_correct"]), bool(p["two_stage_correct"])
        cat = ("both_right" if b and t else
               "2s_wins"    if (not b) and t else
               "base_wins"  if b and (not t) else
               "both_wrong")
        table[(p["level"] or "?", cat)] += 1
        table[("overall", cat)] += 1

    lines = []
    lines.append(r"% Auto-generated win/lose breakdown")
    lines.append(r"\begin{tabular}{l rrrr r}")
    lines.append(r"\toprule")
    lines.append(r"Level & both right & GridProbe wins & baseline wins & both wrong & GP $-$ B \\")
    lines.append(r"\midrule")
    for lv in levels + ["overall"]:
        bw = table[(lv, "both_right")]
        gw = table[(lv, "2s_wins")]
        bbw = table[(lv, "base_wins")]
        bbb = table[(lv, "both_wrong")]
        delta = gw - bbw
        lines.append(f"{lv} & {bw} & {gw} & {bbw} & {bbb} & {delta:+d} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → {out_path}")


def table_per_task_type(per, meta_by_qid, methods, out_path):
    """Group per-question correctness by `third_head` (V2's fine-grained category)."""
    by = {}
    for p in per:
        info = meta_by_qid.get(p["id"], {})
        th = info.get("third_head", "").strip()
        if not th:
            continue
        for m in methods:
            by.setdefault((th, m), [0, 0])
            by[(th, m)][0] += int(p[m + "_correct"])
            by[(th, m)][1] += 1

    types = sorted({k[0] for k in by})
    if not types:
        print("  [skip] no third_head metadata available; per-task-type table not written.")
        return

    pretty = {"baseline_full": "Baseline", "two_stage": "GridProbe", "probe_ensemble": "Probe"}
    show_methods = ["baseline_full", "two_stage", "probe_ensemble"]

    lines = []
    lines.append(r"% Per-task-type (V2 third_head)")
    lines.append(r"\begin{tabular}{l r " + "r " * len(show_methods) + r"r}")
    lines.append(r"\toprule")
    lines.append("Task type & N & " + " & ".join(pretty.get(m, m) for m in show_methods) + r" & GP $-$ B \\")
    lines.append(r"\midrule")
    for th in types:
        n = max(by[(th, m)][1] for m in show_methods)
        cells = []
        for m in show_methods:
            c, t = by[(th, m)]
            cells.append(f"{(c/t*100 if t else 0):.1f}")
        delta = (by[(th, "two_stage")][0] - by[(th, "baseline_full")][0]) / n * 100 if n else 0
        lines.append(f"{th[:50]} & {n} & " + " & ".join(cells) + f" & {delta:+.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → {out_path}")


# ─── Official V2 scoring (verbatim from Video-MME-v2/evaluation/test_video_mme_v2.py) ───

def _cal_relevance(scores):
    """Capability-consistency groups: nonlinear by # correct out of 4."""
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0), correct_count * 25.0


def _cal_logic(scores, group_structure):
    """Reasoning-coherence groups: chain with first-error truncation; the
    score map depends on the group_structure (which 4 questions form chains
    vs. which are siblings)."""
    import ast
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break
    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        return 0.0  # unknown structure
    return score_map.get(last_correct_idx + 1, 0.0)


def _final_rating(method_correct_by_qid, meta_by_qid):
    """Compute V2 official non-linear rating for ONE method.

    method_correct_by_qid: dict {question_id: 0|1}.
    meta_by_qid: dict {question_id: {video_id, group_type, group_structure, level,
                                       second_head, third_head}}

    Returns dict with: total, level_1, level_2, level_3, relevance_score,
    relevance_linear_score, logic_score, second_head_rating, third_head_rating.
    """
    # Build groups: V2's grouping = sorted by question_id, take groups of 4
    qids_sorted = sorted(meta_by_qid.keys())
    groups = []
    cur = []
    for qid in qids_sorted:
        if qid not in method_correct_by_qid:
            continue
        cur.append(qid)
        if len(cur) == 4:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)  # tail

    final = {'level_1': [], 'level_2': [], 'level_3': [],
             'relevance_score': [], 'relevance_linear_score': [],
             'logic_score': [], 'total': []}
    second_head = {}
    third_head = {}

    for grp in groups:
        if len(grp) != 4:
            continue
        last_qid = grp[-1]
        last_meta = meta_by_qid.get(last_qid, {})
        level = last_meta.get('level', '')
        group_type = last_meta.get('group_type', '')
        group_structure = last_meta.get('group_structure', '')
        sh = last_meta.get('second_head', '')
        th = last_meta.get('third_head', '')

        scores = [int(method_correct_by_qid[q]) for q in grp]

        if group_type == 'relevance':
            exp_score, lin_score = _cal_relevance(scores)
            final['relevance_score'].append(exp_score)
            final['relevance_linear_score'].append(lin_score)
        elif group_type == 'logic':
            exp_score = _cal_logic(scores, group_structure)
            final['logic_score'].append(exp_score)
        else:
            continue

        try:
            if level and str(level).strip() and str(level) != 'None':
                lkey = f'level_{int(float(level))}'
                if lkey in final:
                    final[lkey].append(exp_score)
        except (TypeError, ValueError):
            pass
        final['total'].append(exp_score)

        if sh:
            second_head.setdefault(sh, []).append(exp_score)
        if th:
            third_head.setdefault(th, []).append(exp_score)

    out = {k: (sum(v) / len(v) if v else 0.0) for k, v in final.items()}
    out['_n_groups'] = len(groups)
    out['_n_relevance'] = len(final['relevance_score'])
    out['_n_logic'] = len(final['logic_score'])
    out['_second_head'] = {k: (sum(v) / len(v) if v else 0.0) for k, v in second_head.items()}
    out['_third_head']  = {k: (sum(v) / len(v) if v else 0.0) for k, v in third_head.items()}
    return out


def table_nonlin_v2(per, meta_by_qid, methods, out_path):
    """Official Video-MME-v2 grouped non-linear scoring (verbatim from upstream
    `test_video_mme_v2.py`). Outputs total, level-1/2/3, relevance, logic per
    method."""
    pretty = {"baseline_full": "Baseline ($K^2$)",
              "two_stage": "GridProbe (ours)",
              "probe_ensemble": "Probe only"}
    show_methods = ["baseline_full", "two_stage", "probe_ensemble"]

    method_ratings = {}
    for m in show_methods:
        cmap = {p["id"]: int(p[m + "_correct"]) for p in per if p["id"]}
        method_ratings[m] = _final_rating(cmap, meta_by_qid)

    n_groups = method_ratings[show_methods[0]]['_n_groups']
    n_rel    = method_ratings[show_methods[0]]['_n_relevance']
    n_log    = method_ratings[show_methods[0]]['_n_logic']

    lines = []
    lines.append(rf"% Official V2 non-linear grouped score (verbatim from upstream).")
    lines.append(rf"% Groups: total={n_groups}, relevance={n_rel}, logic={n_log}")
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Method & Total & Level 1 & Level 2 & Level 3 & Relevance & Logic \\")
    lines.append(r"\midrule")
    for m in show_methods:
        r = method_ratings[m]
        lines.append(
            f"{pretty.get(m, m)} & {r['total']:.2f} & {r['level_1']:.2f} & "
            f"{r['level_2']:.2f} & {r['level_3']:.2f} & "
            f"{r['relevance_score']:.2f} & {r['logic_score']:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → {out_path}  (groups: total={n_groups}, relevance={n_rel}, logic={n_log})")
    print(f"\n  [Non-linear V2 scores (Total / L1 / L2 / L3 / Rel / Log)]")
    for m in show_methods:
        r = method_ratings[m]
        print(f"    {pretty.get(m, m):<22}  total={r['total']:5.2f}  "
              f"L1={r['level_1']:5.2f}  L2={r['level_2']:5.2f}  L3={r['level_3']:5.2f}  "
              f"rel={r['relevance_score']:5.2f}  log={r['logic_score']:5.2f}")

    # Save second/third head ratings as separate tables
    out_dir = Path(out_path).parent
    for head_name, key in [("second_head", "_second_head"), ("third_head", "_third_head")]:
        # Build matrix: rows = head categories, cols = methods
        cats = sorted(method_ratings[show_methods[0]][key].keys())
        if not cats:
            continue
        head_path = out_dir / f"nonlin_v2_{head_name}.tex"
        head_lines = [rf"% Per-{head_name} non-linear score, official V2 formula"]
        head_lines.append(r"\begin{tabular}{l " + "r " * len(show_methods) + r"r}")
        head_lines.append(r"\toprule")
        head_lines.append(head_name.replace('_', ' ').title() + " & " +
                          " & ".join(pretty.get(m, m) for m in show_methods) + r" & GP $-$ B \\")
        head_lines.append(r"\midrule")
        for cat in cats:
            cells = [f"{method_ratings[m][key].get(cat, 0):.2f}" for m in show_methods]
            delta = method_ratings["two_stage"][key].get(cat, 0) - method_ratings["baseline_full"][key].get(cat, 0)
            head_lines.append(f"{str(cat)[:55]} & " + " & ".join(cells) + f" & {delta:+.2f} \\\\")
        head_lines.append(r"\bottomrule")
        head_lines.append(r"\end{tabular}")
        with open(head_path, "w") as f:
            f.write("\n".join(head_lines))
        print(f"  → {head_path}  ({len(cats)} categories)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to two_stage_eval.py JSON output.")
    p.add_argument("--parquet", default=None,
                   help="V2 test.parquet (enables third_head + non-lin tables).")
    p.add_argument("--out_dir", default="analysis_out")
    p.add_argument("--label", default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.input)
    samples = data.get("results", [])
    print(f"Loaded {len(samples)} samples from {args.input}")
    print(f"Output directory: {out_dir}")

    per, methods = per_sample_correct(samples)

    print("\n[Plots]")
    plot_tflops_distribution(per, out_dir / "tflops_dist.png", label=args.label)
    plot_M_dist(per, out_dir / "M_dist.png", label=args.label)
    plot_tflops_vs_correct(per, out_dir / "tflops_vs_correct.png", label=args.label)

    print("\n[Tables]")
    table_method_x_level(per, methods, out_dir / "method_x_level.tex")
    table_winloss(per, out_dir / "winloss.tex")

    if args.parquet and os.path.exists(args.parquet):
        print(f"\n[Parquet-enriched]")
        print(f"Loading parquet metadata from {args.parquet} ...")
        meta = load_parquet_with_meta(args.parquet)
        table_per_task_type(per, meta, methods, out_dir / "per_task_type.tex")
        table_nonlin_v2(per, meta, methods, out_dir / "nonlin_v2.tex")
    else:
        print("\n[skip] No --parquet provided; per-task-type and non-lin tables not produced.")

    # ── Quick text summary to stdout ──
    print("\n[Quick numerical summary]")
    base_correct = sum(1 for x in per if x["baseline_full_correct"])
    twos_correct = sum(1 for x in per if x["two_stage_correct"])
    n = len(per)
    base_tflops = float(np.mean([x["flops_baseline"]  for x in per])) / 1e12
    twos_tflops = float(np.mean([x["flops_two_stage"] for x in per])) / 1e12
    print(f"  Baseline:   {base_correct}/{n} ({base_correct/n*100:.2f}%)  "
          f"avg TFLOPs={base_tflops:.2f}")
    print(f"  GridProbe:  {twos_correct}/{n} ({twos_correct/n*100:.2f}%)  "
          f"avg TFLOPs={twos_tflops:.2f}  ratio={twos_tflops/base_tflops:.2f}x")
    print(f"  Δ accuracy = {(twos_correct - base_correct)/n*100:+.2f} pp")


if __name__ == "__main__":
    main()
