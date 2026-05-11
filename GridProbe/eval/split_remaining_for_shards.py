"""
Split remaining MDP3-eval work across N GPUs.
=============================================
Reads the existing partial output JSON, computes which sample IDs still need
processing, splits them into N chunks, writes one chunk file per GPU, and
prints ready-to-paste launch commands.

Use this when a sequential mdp3_eval.py run is partway through and you want
to fan out the rest across multiple nodes/GPUs.

Usage:
    python -m GridProbe.eval.split_remaining_for_shards \
        --existing_json results/mdp3_v2_2B_K12_M8.json \
        --benchmark video_mme_v2 \
        --data_dir /storage/Video-MME-v2 \
        --n_chunks 4 \
        --out_dir results/chunks/

Outputs:
    results/chunks/chunk_0.txt ... chunk_N-1.txt   (one ID per line)
    Stdout: launch commands you can paste directly.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_all_ids(benchmark, data_dir, lvb_json, lvb_dir, with_subtitle):
    """Return the ordered list of sample IDs for the chosen benchmark."""
    if benchmark == "video_mme_v2":
        from GridProbe.eval.video_mme_v2 import load_video_mme_v2
        samples = load_video_mme_v2(data_dir, with_subtitle=with_subtitle)
    elif benchmark == "lvb":
        from GridProbe.eval.longvideobench import load_lvb
        samples = load_lvb(lvb_json, lvb_dir=lvb_dir,
                           with_subtitle=with_subtitle)
    else:
        from GridProbe.eval.grid_sampled_ensemble_eval import load_video_mme
        samples = load_video_mme(data_dir)
    return [str(s["id"]) for s in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing_json", required=True,
                    help="Partial mdp3_eval.py output to read 'done' IDs from.")
    ap.add_argument("--benchmark", choices=["video_mme", "video_mme_v2", "lvb"],
                    required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--lvb_json", default=None)
    ap.add_argument("--lvb_dir", default=None)
    ap.add_argument("--with_subtitle", action="store_true")
    ap.add_argument("--n_chunks", type=int, required=True)
    ap.add_argument("--out_dir", default="results/chunks/")

    # Optional: bake the launch-command preamble for the user.
    ap.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--per_sample_M_from", default=None)
    ap.add_argument("--mdp3_lambda", type=float, default=0.2)
    ap.add_argument("--cache_dir", default="/ibex/user/habiam0b/cache_dir")
    ap.add_argument("--output_prefix", default="results/mdp3_shard")

    args = ap.parse_args()

    # ── Load done IDs ──
    with open(args.existing_json) as f:
        existing = json.load(f)
    done = {r["id"] for r in existing.get("results", []) or []}
    print(f"[split] existing JSON: {args.existing_json}")
    print(f"[split] done so far: {len(done)} sample IDs")

    # ── Load full ID list ──
    all_ids = load_all_ids(
        args.benchmark, args.data_dir, args.lvb_json, args.lvb_dir,
        args.with_subtitle)
    print(f"[split] total samples: {len(all_ids)}")

    # ── Compute remaining ──
    remaining = [sid for sid in all_ids if sid not in done]
    print(f"[split] remaining: {len(remaining)}")
    if not remaining:
        print("[split] nothing to do.")
        return

    # ── Split into N chunks (interleaved for balanced level distribution) ──
    chunks = [remaining[i::args.n_chunks] for i in range(args.n_chunks)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        path = out_dir / f"chunk_{i}.txt"
        path.write_text("\n".join(chunk) + "\n")
        chunk_paths.append(path)
        print(f"[split] chunk {i}: {len(chunk)} ids → {path}")

    # ── Print launch commands ──
    base = [
        "python -m GridProbe.eval.mdp3_eval",
        f"  --benchmark {args.benchmark}",
    ]
    if args.benchmark in ("video_mme", "video_mme_v2"):
        base.append(f"  --data_dir {args.data_dir or '<DATA_DIR>'}")
    elif args.benchmark == "lvb":
        base.append(f"  --lvb_json {args.lvb_json or '<LVB_JSON>'}")
        if args.lvb_dir:
            base.append(f"  --lvb_dir {args.lvb_dir}")
    if args.with_subtitle:
        base.append("  --with_subtitle")
    base.extend([
        f"  --vlm_model {args.vlm_model}",
        f"  --K {args.K}",
        f"  --M {args.M}",
        f"  --mdp3_lambda {args.mdp3_lambda}",
        f"  --cache_dir {args.cache_dir}",
        "  --skip_full_baseline",
        # Resume is ON by default (no --no_resume). If a shard crashes,
        # just re-run the same command and it will pick up from the last
        # checkpoint (atomic save every --checkpoint_every samples).
        "  --n_samples 0",
    ])
    if args.per_sample_M_from:
        base.append(f"  --per_sample_M_from {args.per_sample_M_from}")

    print()
    print("=" * 80)
    print(f"  LAUNCH {args.n_chunks} JOBS (one per GPU/node)")
    print("=" * 80)
    for i, cp in enumerate(chunk_paths):
        out = f"{args.output_prefix}_{i}.json"
        cmd_lines = list(base) + [
            f"  --filter_ids_file {cp}",
            f"  --output {out}",
        ]
        print(f"\n# GPU/node {i} (chunk {i}, n={len(chunks[i])} samples):")
        print(" \\\n".join(cmd_lines))

    print()
    print("=" * 80)
    print("  WHEN ALL SHARDS FINISH — merge them:")
    print("=" * 80)
    shard_files = " ".join(f"{args.output_prefix}_{i}.json"
                            for i in range(args.n_chunks))
    print(f"\npython -m GridProbe.eval.merge_shards \\")
    print(f"    --inputs {args.existing_json} {shard_files} \\")
    print(f"    --output {args.output_prefix}_merged.json")
    print()


if __name__ == "__main__":
    main()
