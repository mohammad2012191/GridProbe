"""
Golden-Cell Importance Map Visualization
==========================================
Run row + col passes on a K×K grid, compute per-axis confidence, then build
a 2D importance map M[r,c] and save heatmaps.

Uses PIL only — no matplotlib dependency.

For each video:
  1. Extract K² frames, arrange conceptually as K×K grid
  2. K row passes → row_conf[r] (confidence of each row)
  3. K col passes → col_conf[c] (confidence of each column)
  4. M[r,c] = row_conf[r] * col_conf[c]  (multiplicative)
  5. Save: heatmap PNG, golden-cell thumbnails, JSON with raw numbers

Usage:
    python -m GridProbe.eval.importance_map \
        --data_dir /path/to/Video-MME \
        --n_per_bin 2 --K 8 --debug
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GridProbe.eval.grid_sampled_ensemble_eval import (
    extract_frames_uniform, build_inputs, score_letters_for_pass,
    get_letter_token_id_variants, row_indices, col_indices,
    load_video_mme,
)
from GridProbe.eval.video_mme import build_prompt

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D"]


# ═══════════════════════════════════════════════════════════════
# PIL-based drawing helpers (zero matplotlib)
# ═══════════════════════════════════════════════════════════════

def _get_font(size=12):
    """Try to get a TTF font; fall back to default bitmap."""
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def val_to_rgb(val, vmin, vmax):
    """Map a scalar to a yellow→orange→red color (like YlOrRd)."""
    t = (val - vmin) / (vmax - vmin + 1e-10)
    t = max(0.0, min(1.0, t))
    # Yellow (255,255,200) → Orange (255,140,0) → Red (180,0,0)
    if t < 0.5:
        s = t / 0.5
        r = 255
        g = int(255 - (255 - 140) * s)
        b = int(200 - 200 * s)
    else:
        s = (t - 0.5) / 0.5
        r = int(255 - (255 - 180) * s)
        g = int(140 - 140 * s)
        b = 0
    return (r, g, b)


def draw_heatmap(M, K, cell_size=60, title="", annotations=None, golden_cells=None):
    """Draw a K×K heatmap as a PIL Image.

    annotations: list of (r, c, text) to draw in each cell
    golden_cells: list of (r, c) to highlight with cyan border
    """
    font = _get_font(11)
    font_sm = _get_font(9)
    title_font = _get_font(14)

    margin_top = 30
    margin_left = 40
    margin_bottom = 25
    margin_right = 50  # for colorbar
    w = margin_left + K * cell_size + margin_right
    h = margin_top + K * cell_size + margin_bottom

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    vmin, vmax = float(M.min()), float(M.max())

    # Title
    if title:
        draw.text((margin_left, 4), title, fill=(0, 0, 0), font=title_font)

    # Cells
    for r in range(K):
        for c in range(K):
            x0 = margin_left + c * cell_size
            y0 = margin_top + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            color = val_to_rgb(M[r, c], vmin, vmax)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(100, 100, 100))

            # Frame index
            frame_idx = r * K + c
            brightness = sum(color) / 3
            txt_color = (255, 255, 255) if brightness < 160 else (0, 0, 0)
            draw.text((x0 + 4, y0 + 3), f"f{frame_idx}", fill=txt_color, font=font_sm)

    # Annotations (e.g., golden cell labels)
    if annotations:
        for r, c, text in annotations:
            x0 = margin_left + c * cell_size
            y0 = margin_top + r * cell_size
            draw.text((x0 + 4, y0 + cell_size // 2), text, fill=(0, 200, 255), font=font)

    # Golden cell borders
    if golden_cells:
        for rank, (r, c) in enumerate(golden_cells):
            x0 = margin_left + c * cell_size
            y0 = margin_top + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            for offset in range(3):
                draw.rectangle([x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                               outline=(0, 220, 255))
            draw.text((x0 + cell_size // 3, y0 + cell_size // 2 + 10),
                      f"#{rank+1}", fill=(0, 220, 255), font=font)

    # Row/col labels
    for r in range(K):
        y = margin_top + r * cell_size + cell_size // 3
        draw.text((2, y), f"R{r}", fill=(80, 80, 80), font=font_sm)
    for c in range(K):
        x = margin_left + c * cell_size + cell_size // 4
        draw.text((x, margin_top + K * cell_size + 4), f"C{c}", fill=(80, 80, 80), font=font_sm)

    # Simple colorbar
    bar_x = margin_left + K * cell_size + 10
    bar_w = 15
    bar_h = K * cell_size
    for py in range(bar_h):
        t = 1.0 - py / bar_h  # top = max
        v = vmin + t * (vmax - vmin)
        color = val_to_rgb(v, vmin, vmax)
        draw.line([(bar_x, margin_top + py), (bar_x + bar_w, margin_top + py)], fill=color)
    draw.text((bar_x, margin_top - 12), f"{vmax:.2f}", fill=(0, 0, 0), font=font_sm)
    draw.text((bar_x, margin_top + bar_h + 2), f"{vmin:.2f}", fill=(0, 0, 0), font=font_sm)

    return img


def draw_conf_bars(confs, preds, answer, K, axis_name="Row", bar_width=300, bar_height=22):
    """Draw a horizontal bar chart of per-pass confidence as a PIL Image."""
    font = _get_font(11)
    font_sm = _get_font(9)
    title_font = _get_font(13)

    margin_left = 80
    margin_top = 25
    margin_right = 20
    w = margin_left + bar_width + margin_right
    h = margin_top + K * bar_height + 10

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((4, 4), f"{axis_name} confidence (green=correct)", fill=(0, 0, 0), font=title_font)

    cmax = max(confs) if max(confs) > 0 else 1.0

    for i in range(K):
        y = margin_top + i * bar_height
        bar_len = int(confs[i] / cmax * bar_width)
        color = (80, 180, 80) if preds[i] == answer else (220, 120, 120)
        draw.rectangle([margin_left, y + 2, margin_left + bar_len, y + bar_height - 2],
                       fill=color, outline=(100, 100, 100))
        label = f"{axis_name[0]}{i}→{preds[i]}"
        draw.text((4, y + 3), label, fill=(0, 0, 0), font=font)
        draw.text((margin_left + bar_len + 4, y + 3), f"{confs[i]:.2f}", fill=(80, 80, 80), font=font_sm)

    return img


def compose_figure(heatmap_img, row_bar_img, col_bar_img, thumb_strip, info_text, out_path):
    """Stack all panels into one final PNG."""
    font = _get_font(12)

    # Info text panel
    info_w = max(heatmap_img.width, 600)
    info_h = 70
    info_img = Image.new("RGB", (info_w, info_h), (245, 245, 245))
    draw = ImageDraw.Draw(info_img)
    lines = info_text.split("\n")
    for i, line in enumerate(lines):
        draw.text((8, 4 + i * 16), line, fill=(0, 0, 0), font=font)

    # Compose: info | heatmap + bars side by side | thumbnails
    bars_combined_w = max(row_bar_img.width, col_bar_img.width)
    bars_combined_h = row_bar_img.height + col_bar_img.height + 10
    bars_img = Image.new("RGB", (bars_combined_w, bars_combined_h), (255, 255, 255))
    bars_img.paste(row_bar_img, (0, 0))
    bars_img.paste(col_bar_img, (0, row_bar_img.height + 10))

    top_row_w = heatmap_img.width + bars_img.width + 20
    top_row_h = max(heatmap_img.height, bars_img.height)
    top_row = Image.new("RGB", (top_row_w, top_row_h), (255, 255, 255))
    top_row.paste(heatmap_img, (0, 0))
    top_row.paste(bars_img, (heatmap_img.width + 20, 0))

    total_w = max(info_w, top_row_w, thumb_strip.width if thumb_strip else 0)
    total_h = info_h + top_row_h + (thumb_strip.height if thumb_strip else 0) + 20

    final = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    final.paste(info_img, (0, 0))
    final.paste(top_row, (0, info_h + 5))
    if thumb_strip:
        final.paste(thumb_strip, (0, info_h + top_row_h + 15))

    final.save(out_path)
    logger.info("Saved %s", out_path)


def make_thumb_strip(frames, golden_indices, K, M_mult, thumb_size=120):
    """Create a horizontal strip of golden-cell frame thumbnails."""
    n = min(len(golden_indices), 5)
    if n == 0:
        return None

    font = _get_font(10)
    pad = 8
    strip_w = n * (thumb_size + pad) + pad
    strip_h = thumb_size + 40

    strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    for i in range(n):
        r, c = golden_indices[i]
        frame_idx = r * K + c
        if frame_idx >= len(frames):
            continue

        thumb = frames[frame_idx].copy()
        thumb.thumbnail((thumb_size, thumb_size))
        x = pad + i * (thumb_size + pad)
        y = 0
        strip.paste(thumb, (x, y))

        # Border
        draw.rectangle([x - 1, y - 1, x + thumb.width, y + thumb.height],
                       outline=(0, 220, 255), width=2)
        # Label
        imp = M_mult[r, c]
        draw.text((x, y + thumb.height + 2),
                  f"#{i+1} f{frame_idx} ({r},{c}) {imp:.3f}",
                  fill=(0, 0, 0), font=font)

    return strip


# ═══════════════════════════════════════════════════════════════
# Core logic (unchanged from before)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_axis_passes(model, processor, prompt_text, all_frames, axis_fn, K,
                    letter_variants, device, max_video_pixels):
    """Run all passes for one axis. Returns (lp_matrix, confs, preds)."""
    indices_list = axis_fn(K)
    lp_list = []

    for indices in indices_list:
        subset = [all_frames[i] for i in indices]
        lp = score_letters_for_pass(
            model, processor, prompt_text, subset,
            letter_variants, device, max_video_pixels)
        lp_list.append(lp.cpu())

    lp_matrix = torch.stack(lp_list)
    probs = F.softmax(lp_matrix, dim=-1)
    confs = probs.max(dim=-1).values.numpy()
    preds = [LETTERS[i] for i in lp_matrix.argmax(dim=-1).tolist()]
    return lp_matrix, confs, preds


def build_importance_map(row_confs, col_confs, K):
    """Build K×K importance map from marginal confidences."""
    rc = row_confs[:K]
    cc = col_confs[:K]
    M_add = rc[:, None] + cc[None, :]
    M_mult = rc[:, None] * cc[None, :]
    return M_add, M_mult


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--vlm_model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--K", type=int, default=8, help="Grid side length (K²=total frames)")
    p.add_argument("--max_video_pixels", type=int, default=0)
    p.add_argument("--n_per_bin", type=int, default=2,
                   help="Number of videos per duration bin (short/medium/long)")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_dir", default="importance_maps")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    os.makedirs(args.out_dir, exist_ok=True)
    K = args.K
    n_frames = K * K

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

    # ── Pick videos: n_per_bin from each duration bin ──
    all_samples = load_video_mme(args.data_dir)
    selected = []
    for bin_name in ["short", "medium", "long"]:
        bin_samples = [s for s in all_samples if s["duration_bin"] == bin_name]
        selected.extend(bin_samples[:args.n_per_bin])
    logger.info("Selected %d videos (%d per bin)", len(selected), args.n_per_bin)

    all_results = []

    for si, sample in enumerate(selected):
        vid_label = f"{sample['duration_bin']}_{sample['id'][:12]}"
        logger.info("[%d/%d] Processing %s ...", si + 1, len(selected), vid_label)

        frames = extract_frames_uniform(sample["video_path"], n_frames)
        if frames is None:
            logger.warning("  Frame extraction failed, skipping")
            continue

        prompt_text = build_prompt(sample["question"], sample["options"])
        answer = sample["answer"]

        try:
            # ── Row passes ──
            row_lp, row_confs, row_preds = run_axis_passes(
                model, processor, prompt_text, frames, row_indices, K,
                letter_variants, device, args.max_video_pixels)
            logger.info("  Rows: preds=%s  confs=%s",
                        "".join(row_preds), np.round(row_confs, 2))

            # ── Col passes ──
            col_lp, col_confs, col_preds = run_axis_passes(
                model, processor, prompt_text, frames, col_indices, K,
                letter_variants, device, args.max_video_pixels)
            logger.info("  Cols: preds=%s  confs=%s",
                        "".join(col_preds), np.round(col_confs, 2))

            # ── Importance maps ──
            M_add, M_mult = build_importance_map(row_confs, col_confs, K)

            # ── Golden cells (top-5 by multiplicative importance) ──
            flat = M_mult.flatten()
            top5_flat = np.argsort(flat)[-5:][::-1]
            golden = [(int(idx // K), int(idx % K)) for idx in top5_flat]

            # ── Ensemble ──
            all_lp = torch.cat([row_lp, col_lp], dim=0)
            ensemble_pred = LETTERS[all_lp.mean(dim=0).argmax().item()]

            from collections import Counter
            row_majority = Counter(row_preds).most_common(1)[0][0]
            col_majority = Counter(col_preds).most_common(1)[0][0]

            result = {
                "id": sample["id"],
                "duration_bin": sample["duration_bin"],
                "question": sample["question"][:100],
                "answer": answer,
                "row_preds": row_preds,
                "col_preds": col_preds,
                "row_confs": row_confs.tolist(),
                "col_confs": col_confs.tolist(),
                "row_majority": row_majority,
                "col_majority": col_majority,
                "ensemble_pred": ensemble_pred,
                "golden_cells": golden,
                "correct_row_majority": row_majority == answer,
                "correct_col_majority": col_majority == answer,
                "correct_ensemble": ensemble_pred == answer,
            }
            all_results.append(result)

            logger.info("  ans=%s  row_maj=%s(%s)  col_maj=%s(%s)  ens=%s(%s)  golden=%s",
                        answer,
                        row_majority, "OK" if row_majority == answer else "XX",
                        col_majority, "OK" if col_majority == answer else "XX",
                        ensemble_pred, "OK" if ensemble_pred == answer else "XX",
                        golden[:3])

            # ── Draw & save ──
            annotations = [(r, c, f"#{rank+1}") for rank, (r, c) in enumerate(golden)]
            heatmap = draw_heatmap(M_mult, K, cell_size=60,
                                   title="Importance (row_conf * col_conf)",
                                   golden_cells=golden)
            row_bars = draw_conf_bars(row_confs, row_preds, answer, K, "Row")
            col_bars = draw_conf_bars(col_confs, col_preds, answer, K, "Col")
            thumbs = make_thumb_strip(frames, golden, K, M_mult)

            info_text = (
                f"[{sample['duration_bin']}] {sample['id'][:25]}   Answer: {answer}\n"
                f"Q: {sample['question'][:90]}...\n"
                f"Row majority: {row_majority}  Col majority: {col_majority}  "
                f"Ensemble: {ensemble_pred}  "
                f"Options: {' | '.join(sample.get('options', []))[:120]}"
            )

            out_path = os.path.join(args.out_dir, f"{vid_label}.png")
            compose_figure(heatmap, row_bars, col_bars, thumbs, info_text, out_path)

        except Exception as e:
            logger.warning("  Failed: %s", e)
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  IMPORTANCE MAP SUMMARY")
    print("=" * 70)
    for r in all_results:
        tag = f"[{r['duration_bin']}] {r['id'][:15]}"
        print(f"  {tag:<30}  ans={r['answer']}  "
              f"row={r['row_majority']}({'OK' if r['correct_row_majority'] else 'XX'})  "
              f"col={r['col_majority']}({'OK' if r['correct_col_majority'] else 'XX'})  "
              f"ens={r['ensemble_pred']}({'OK' if r['correct_ensemble'] else 'XX'})  "
              f"top3={r['golden_cells'][:3]}")

    n = len(all_results)
    if n > 0:
        print(f"\n  Accuracy: row_maj={sum(r['correct_row_majority'] for r in all_results)}/{n}  "
              f"col_maj={sum(r['correct_col_majority'] for r in all_results)}/{n}  "
              f"ensemble={sum(r['correct_ensemble'] for r in all_results)}/{n}")

    json_path = os.path.join(args.out_dir, "importance_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {json_path}")
    print(f"  Heatmaps in {args.out_dir}/")


if __name__ == "__main__":
    main()
