"""
LongVideoBench loader / prompt / answer-check for two_stage_eval.

LVB schema (from upstream lvb_val.json):
    video_id           : str  (also used as basename for video file)
    id                 : str  (question id; usually <video_id>_<idx>)
    question           : str
    candidates         : list[str]  (typically 4 options)
    correct_choice     : int   (0-indexed into candidates)
    subtitle_path      : str   (relative to lvb_dir/subtitles/)
    duration_group     : int   (15/60/600/3600  -- duration bucket in seconds)
    duration           : float (actual video duration in seconds)
    position           : list[int]  (subtitle/word positions where evidence sits)
    starting_timestamp_for_subtitles : int
    level              : str   ("L1-..." / "L2-..." / etc.)
    topic_category, question_category : str

We expose the same sample-dict shape that two_stage_eval expects from V1/V2:
    id, duration_bin, video_path, question, options, answer, subtitle
"""

import json
import os
from pathlib import Path

# LVB has 4 candidates per question (sometimes 5 in older releases);
# we use A..E to be safe and dynamically map per-question via len(candidates).
LETTERS_LVB = ["A", "B", "C", "D", "E"]

# Bins for stratification: we use the LVB duration_group as-is.
# 15 / 60 / 600 / 3600 seconds  →  short / med / long / very-long
BINS_LVB = ["15", "60", "600", "3600"]


def _resolve_video_path(lvb_dir, video_id):
    """Find the actual video file given an LVB video_id."""
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        cand = os.path.join(lvb_dir, "videos", video_id + ext)
        if os.path.exists(cand):
            return cand
    # Also try root and subdirs
    p = Path(lvb_dir) / "videos"
    if p.exists():
        for f in p.glob(f"{video_id}.*"):
            return str(f)
    return os.path.join(lvb_dir, "videos", video_id + ".mp4")  # fallback (may not exist)


def _load_subtitle_text(lvb_dir, item, max_chars=3000):
    """Load and lightly format LVB subtitle JSON into a single text blob."""
    sub_path = item.get("subtitle_path", "")
    if not sub_path:
        return ""
    full = os.path.join(lvb_dir, sub_path) if not os.path.isabs(sub_path) else sub_path
    if not os.path.exists(full):
        # try under subtitles/
        alt = os.path.join(lvb_dir, "subtitles", os.path.basename(sub_path))
        full = alt if os.path.exists(alt) else full
    if not os.path.exists(full):
        return ""
    try:
        with open(full, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Plain text fallback
        try:
            with open(full, "r", encoding="utf-8") as f:
                return f.read().strip()[:max_chars]
        except Exception:
            return ""
    # LVB subtitles JSON is a list of {start, end, text} dicts
    if isinstance(data, list):
        parts = []
        for seg in data:
            t = seg.get("text") or seg.get("line") or ""
            if t:
                parts.append(t.strip())
        text = " ".join(parts)
    elif isinstance(data, dict):
        text = data.get("text", str(data))
    else:
        text = str(data)
    return text[:max_chars]


def load_lvb(lvb_json_or_dir, lvb_dir=None, with_subtitle=False):
    """
    Load LongVideoBench samples.

    Args:
        lvb_json_or_dir : path to either an LVB split JSON (e.g. lvb_val.json)
                          or to the LVB root dir (in which case we look for lvb_val.json).
        lvb_dir         : root dir for resolving video_path / subtitle_path.
                          If None, inferred from lvb_json_or_dir's parent.
        with_subtitle   : if True, include subtitle text in returned samples.

    Returns:
        list[dict] with keys: id, duration_bin, video_path, question, options,
                              answer, subtitle (optional)
    """
    p = Path(lvb_json_or_dir)
    if p.is_dir():
        json_path = p / "lvb_val.json"
        root = p
    else:
        json_path = p
        root = p.parent

    if lvb_dir is not None:
        root = Path(lvb_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for it in raw:
        video_id = it.get("video_id") or it.get("id", "").split("_")[0]
        candidates = it.get("candidates") or []
        cc = it.get("correct_choice", 0)
        if not isinstance(candidates, list) or len(candidates) == 0:
            continue
        # LVB stores correct_choice as int index; convert to letter
        try:
            answer_letter = LETTERS_LVB[int(cc)]
        except (ValueError, IndexError):
            continue
        sample = {
            "id":            it.get("id") or f"{video_id}_0",
            "duration_bin":  str(it.get("duration_group", "0")),
            "video_path":    _resolve_video_path(str(root), video_id),
            "question":      it.get("question", ""),
            "options":       list(candidates),
            "answer":        answer_letter,
            # extras for analysis
            "video_id":      video_id,
            "_lvb_position": it.get("position", []),
            "_lvb_level":    it.get("level", ""),
            "_lvb_topic":    it.get("topic_category", ""),
            "_lvb_qcat":     it.get("question_category", ""),
        }
        # Carry over the strat helpers if this JSON came from lvb_stratified_sample
        for k in ("__strat_bin", "__metric_name", "__metric_value",
                  "__n_positions", "__span"):
            if k in it:
                sample[k] = it[k]
        if with_subtitle:
            sample["subtitle"] = _load_subtitle_text(str(root), it)
        samples.append(sample)
    return samples


def build_prompt_lvb(question, options, subtitle=None):
    """LVB MC prompt; mirrors the upstream format used by lvb_eval.py."""
    opts = "\n".join(f"({LETTERS_LVB[i]}) {c}" for i, c in enumerate(options))
    if subtitle:
        return (f"Video subtitles:\n{subtitle}\n\n"
                f"Question: {question}\n\n{opts}\n\n"
                f"Answer with the letter of the correct option.")
    return (f"Question: {question}\n\n{opts}\n\n"
            f"Answer with the letter of the correct option.")


def check_answer_lvb(predicted, answer_letter, options=None):
    """
    Check if the predicted letter matches the gold letter.
    Accepts a logprob-decoded single letter (the common path in two_stage_eval),
    or a free-form string (the lvb_eval.py path).
    """
    import re
    if not predicted:
        return False
    pred = str(predicted).strip()
    # Fast path: already a single letter
    if len(pred) == 1 and pred.upper() in LETTERS_LVB:
        return pred.upper() == answer_letter
    # Strip "Answer:" / "(C)" / "C." style prefixes
    pred = re.sub(r"^(?:answer\s*[:is]*\s*)", "", pred, flags=re.IGNORECASE).strip()
    pu = pred.upper()
    m = re.match(r"^\(?([A-E])\)?[\.\s)]", pu)
    if m:
        return m.group(1) == answer_letter
    if len(pu) <= 3:
        m = re.match(r"^\(?([A-E])\)?$", pu)
        if m:
            return m.group(1) == answer_letter
    m = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", pu)
    if m:
        return m.group(1) == answer_letter
    return False
