"""
Probe: does Qwen3-VL crush per-frame tokens when you ask for many frames?

For each (n_frames) setting, picks a long video, runs the processor, and
reports:
  • total visual tokens (before spatial merge)  = T * H * W
  • tokens per frame                            = (H * W)
  • visual tokens after spatial merge           = (total / merge²)
  • LM sequence length                          = input_ids.shape[1]

If tokens-per-frame drops as n_frames grows, Qwen is crushing resolution
to fit the context → our capability-extension story holds.

Usage:
    python -m GridProbe.eval.probe_compression \
        --data_dir /path/to/Video-MME \
        --frame_counts 64,256,512,1024,2048
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, load_video_mme,
)
from GridProbe.eval.video_mme import build_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--frame_counts", default="64,256,512,1024,2048")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--duration_bin", default="long")
    args = p.parse_args()

    from transformers import AutoProcessor
    proc_kw = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    processor = AutoProcessor.from_pretrained(args.vlm_model, **proc_kw)

    # Pick one long video
    samples = load_video_mme(args.data_dir)
    if args.duration_bin:
        samples = [s for s in samples if s["duration_bin"] == args.duration_bin]
    assert samples, "No samples found"
    s = samples[0]
    prompt = build_prompt(s["question"], s["options"])

    counts = [int(x) for x in args.frame_counts.split(",") if x.strip()]
    max_needed = max(counts)

    print(f"Probing video: {os.path.basename(s['video_path'])}")
    print(f"Extracting {max_needed} frames (will subsample)...")
    frames_all = extract_frames_uniform(s["video_path"], max_needed)
    assert frames_all is not None, "Frame extraction failed"

    import numpy as np

    print()
    print(f"  {'n_frames':>9} {'T_grid':>8} {'H':>5} {'W':>5} {'tok/frame':>10} {'total_vis':>11} {'LM seq':>9}")
    print("  " + "-" * 60)

    for nf in counts:
        idx = np.linspace(0, len(frames_all) - 1, nf).round().astype(int).tolist()
        frames = [frames_all[i] for i in idx]
        try:
            inputs = build_inputs(processor, prompt, frames, max_video_pixels=0)
        except Exception as e:
            print(f"  {nf:>9}  FAILED: {e}")
            continue

        vgt = inputs["video_grid_thw"]
        T, H, W = int(vgt[0, 0]), int(vgt[0, 1]), int(vgt[0, 2])
        tok_per_frame = H * W
        total_visual = T * H * W
        lm_seq = int(inputs["input_ids"].shape[1])
        print(f"  {nf:>9} {T:>8} {H:>5} {W:>5} {tok_per_frame:>10} {total_visual:>11} {lm_seq:>9}")

    print()
    print("Interpretation:")
    print("  If tok/frame (H*W) DROPS as n_frames grows → Qwen is crushing resolution")
    print("    → our capability-extension story holds: grid can keep full-res per frame")
    print("  If tok/frame STAYS CONSTANT → processor isn't compressing; context must blow up")
    print("    → check if LM seq grows linearly; if so grid's context-factorization still wins")


if __name__ == "__main__":
    main()
