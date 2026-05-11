"""
Video-MME Evaluation (Option B)
================================
Evaluates on Video-MME using the FULL Qwen3-VL model natively.

GridFormer acts as a frame selector/organizer BEFORE the VLM, not inside it.
The pre-trained vision↔language alignment in Qwen3-VL stays intact.

Usage (standalone):
    python -m GridFormer.eval.video_mme \
        --video_mme_dir /path/to/video_mme \
        --n_samples 120 \
        --n_frames 64 \
        --debug

Reports accuracy split by video duration (Short / Medium / Long).
"""

import os
import re
import sys
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ─── Shared prompt template ────────────────────────────────────────────────
PROMPT_TEMPLATE_MC = """Question: {question}

{options}

Answer with the letter of the correct option (A, B, C, or D)."""

PROMPT_TEMPLATE_OPEN = """Question: {question}

Answer briefly."""


def build_prompt(question: str, options: List[str]) -> str:
    """Build a prompt for video QA — shared between training and evaluation."""
    if options:
        valid = [o for o in options if o and str(o).strip()]
        if valid:
            opts_text = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(valid))
            return PROMPT_TEMPLATE_MC.format(question=question, options=opts_text)
    return PROMPT_TEMPLATE_OPEN.format(question=question)


def check_answer(predicted: str, correct: str, options: List[str] = None) -> bool:
    """Extract the answer from model output and check correctness.

    Priority:
      1. \\boxed{X}  (legacy / if model still produces it)
      2. Standalone letter on its own line or at the very start
      3. First A-D letter found anywhere in the text
    """
    correct = str(correct).strip().upper()
    predicted = str(predicted).strip()

    if len(correct) != 1 or correct not in "ABCD":
        return predicted.upper() == correct.upper()

    # 1. \boxed{X} (still accept it if present)
    boxed = re.findall(r'\\boxed\{([A-Da-d])\}', predicted)
    if boxed:
        return boxed[-1].upper() == correct

    # 2. Strip common prefixes: "Answer:", "The answer is", etc.
    cleaned = re.sub(r'^(?:answer\s*[:is]*\s*)', '', predicted, flags=re.IGNORECASE).strip()

    # 3. Standalone letter at start: "(C)" or "C." or "C "
    head = cleaned[:5].strip().upper()
    m = re.match(r'^\(?([A-D])\)?[.\s]*$', head)
    if m:
        return m.group(1) == correct

    # 4. First isolated A-D letter (skip letters inside words like "Answer")
    m = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', cleaned.upper())
    if m:
        return m.group(1) == correct

    return False


class VideoMMEEvaluator:
    """
    Evaluates on Video-MME using native Qwen3-VL inference.
    
    Usage:
        evaluator = VideoMMEEvaluator(video_mme_dir)
        results = evaluator.evaluate(model, processor, n_frames=64, ...)
    """
    
    def __init__(
        self,
        video_mme_dir: str,
        grid_k: int = 8,
        sub_grid_k: int = 4,
        cell_size: int = 336,
    ):
        self.video_mme_dir = video_mme_dir
        self.grid_k = grid_k
        self.sub_grid_k = sub_grid_k
        self.cell_size = cell_size
        self.samples = self._load_video_mme()
    
    @staticmethod
    def _strip_option_prefix(opt: str) -> str:
        """Strip letter prefix like 'A. ' from option text."""
        opt = opt.strip()
        if len(opt) >= 3 and opt[0] in "ABCD" and opt[1] in ". )":
            return opt[2:].strip()
        if len(opt) >= 4 and opt[0] == "(" and opt[1] in "ABCD" and opt[2] == ")":
            return opt[3:].strip()
        return opt
    
    def _load_video_mme(self) -> List[Dict]:
        """Load Video-MME from parquet."""
        samples = []
        parquet_path = os.path.join(self.video_mme_dir, "test-00000-of-00001.parquet")
        if os.path.exists(parquet_path):
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                logger.info(f"Video-MME parquet: {df.shape[0]} rows, columns: {list(df.columns)}")
                
                for _, row in df.iterrows():
                    youtube_id = str(row.get("videoID", ""))
                    raw_options = row.get("options", [])
                    if isinstance(raw_options, (list, np.ndarray)):
                        options_clean = [self._strip_option_prefix(str(o)) for o in raw_options]
                    else:
                        options_clean = []
                    
                    answer_letter = str(row.get("answer", "")).strip().upper()
                    dur = str(row.get("duration", "medium")).strip().lower()
                    
                    samples.append({
                        "video_id": youtube_id,
                        "question": str(row.get("question", "")),
                        "question_id": str(row.get("question_id", "")),
                        "options": options_clean,
                        "answer": answer_letter,
                        "duration_bin": dur,
                        "task_type": str(row.get("task_type", "")),
                    })
            except ImportError:
                logger.error("pandas required for parquet. pip install pandas pyarrow")
                return []
        else:
            logger.warning(f"No parquet at {parquet_path}")
            return []
        
        logger.info(f"Video-MME: {len(samples)} samples loaded")
        bins = {}
        for s in samples:
            b = s["duration_bin"]
            bins[b] = bins.get(b, 0) + 1
        logger.info(f"  Duration split: {bins}")
        
        return samples
    
    def _find_video(self, video_id: str) -> Optional[str]:
        """Find video file by YouTube ID."""
        for subdir in ["videos/data", "videos", "data", "ytb_videos", "."]:
            dirpath = os.path.join(self.video_mme_dir, subdir)
            if not os.path.isdir(dirpath):
                continue
            for ext in [".mp4", ".mkv", ".avi", ".webm"]:
                path = os.path.join(dirpath, video_id + ext)
                if os.path.exists(path):
                    return path
        return None
    
    @torch.no_grad()
    def evaluate(
        self,
        model,
        processor,
        n_samples: int = -1,
        n_frames: int = 64,
        device: torch.device = None,
        debug: bool = False,
    ) -> Dict:
        """
        Evaluate on Video-MME using native Qwen3-VL inference.
        
        Args:
            model: Full Qwen3-VL model (vision + language intact)
            processor: Qwen3-VL processor
            n_samples: Number of samples (-1 = all)
            n_frames: Frames per video to feed to VLM
            device: GPU device
            debug: Verbose logging
            
        Returns:
            dict with "short", "medium", "long", "overall" accuracy percentages
            + "detailed" per-sample results
        """
        from qwen_vl_utils import process_vision_info
        
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        # ── Stratified sampling ──
        samples = self.samples
        if n_samples > 0:
            short_s = [s for s in samples if s.get("duration_bin") == "short"]
            med_s = [s for s in samples if s.get("duration_bin") == "medium"]
            long_s = [s for s in samples if s.get("duration_bin") == "long"]
            per_bin = max(1, n_samples // 3)
            samples = short_s[:per_bin] + med_s[:per_bin] + long_s[:per_bin]
        
        logger.info(f"Evaluating {len(samples)} samples with {n_frames} frames/video")
        
        results_by_bin = {"short": [], "medium": [], "long": []}
        detailed = []
        
        for sample in tqdm(samples, desc="Video-MME eval", disable=not debug):
            video_id = sample["video_id"]
            question = sample["question"]
            options = sample["options"]
            correct_answer = sample["answer"]
            dur_bin = sample["duration_bin"]
            
            video_path = self._find_video(video_id)
            if video_path is None:
                continue
            
            try:
                prompt_text = build_prompt(question, options)
                
                # ── Native Qwen3-VL video inference ──
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "nframes": n_frames,
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        min_p=0,
                    )
                
                # Trim input tokens
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                pred_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0].strip()
                
                is_correct = check_answer(pred_text, correct_answer, options)
                results_by_bin[dur_bin].append(is_correct)
                
                detailed.append({
                    "video_id": video_id,
                    "duration_bin": dur_bin,
                    "correct": is_correct,
                    "predicted": pred_text,
                    "answer": correct_answer,
                })
                
                if debug:
                    mark = "✓" if is_correct else "✗"
                    # Show just the final answer portion
                    boxed = re.findall(r'\\boxed\{([A-Da-d])\}', pred_text)
                    extracted = boxed[-1].upper() if boxed else pred_text[:30]
                    logger.info(f"  {mark} [{dur_bin}] {video_id}: extracted={extracted} ans={correct_answer}")
                
            except Exception as e:
                if debug:
                    logger.warning(f"Error evaluating {video_id}: {e}")
                continue
        
        # ── Compute accuracies ──
        accs = {}
        all_correct = []
        for bin_name in ["short", "medium", "long"]:
            if results_by_bin[bin_name]:
                acc = 100 * np.mean(results_by_bin[bin_name])
                accs[bin_name] = round(acc, 2)
                all_correct.extend(results_by_bin[bin_name])
            else:
                accs[bin_name] = 0.0
        
        accs["overall"] = round(100 * np.mean(all_correct), 2) if all_correct else 0.0
        accs["n_evaluated"] = len(all_correct)
        accs["detailed"] = detailed
        
        return accs


def format_eval_results(results: Dict) -> str:
    """Format evaluation results for logging."""
    lines = [
        "┌─────────────────────────────────┐",
        "│     Video-MME Evaluation        │",
        "├─────────────────────────────────┤",
        f"│  Short:   {results.get('short', 0):6.2f}%              │",
        f"│  Medium:  {results.get('medium', 0):6.2f}%              │",
        f"│  Long:    {results.get('long', 0):6.2f}%              │",
        f"│  Overall: {results.get('overall', 0):6.2f}%              │",
        f"│  N:       {results.get('n_evaluated', 0):>6d}               │",
        "└─────────────────────────────────┘",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on Video-MME (Option B)")
    parser.add_argument("--video_mme_dir", required=True, help="Path to Video-MME data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--n_frames", type=int, default=64)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    logger.info(f"Loading {args.model}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    logger.info("Model loaded.")
    
    evaluator = VideoMMEEvaluator(args.video_mme_dir)
    results = evaluator.evaluate(
        model, processor,
        n_samples=args.n_samples,
        n_frames=args.n_frames,
        debug=args.debug,
    )
    
    print(format_eval_results(results))
    
    # Save detailed results
    output_path = args.output or os.path.join(args.video_mme_dir, "gridformer_eval_results.json")
    results_to_save = {k: v for k, v in results.items() if k != "detailed"}
    results_to_save["detailed"] = results.get("detailed", [])
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {output_path}")
