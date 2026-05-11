"""
Video-MME-v2 utilities (8-option MCQ, A–H).

Differences from Video-MME v1:
  - 8 options (A through H) instead of 4
  - Different parquet schema:
        video_id, url, group_type ('logic' | 'relevance'), group_structure,
        question_id, question, options (single string), answer (A-H),
        level (1, 2, 3 for some questions, null for others within a group),
        second_head, third_head
  - Options are encoded as ONE string e.g. "A. Foo. B. Bar. C. Baz."
    so we parse them into a list[str] for our prompt builder.
  - Each video has 4 questions grouped together; only the 4th question in
    each group has a non-null `level`.

Public API:
    LETTERS_V2:                list of "A"..."H"
    parse_options_v2(s):       split the option string into a list of choices
    build_prompt_v2(...):      construct an MCQ prompt
    check_answer_v2(...):      lenient MCQ answer extraction
    load_video_mme_v2(...):    return list of sample dicts compatible with
                               our existing eval pipeline
"""

import os
import re
from pathlib import Path
from typing import List, Optional


LETTERS_V2 = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ─────────────────────────────────────────────────────────────────────
# Option parsing
# ─────────────────────────────────────────────────────────────────────

_OPT_SPLIT_RE = re.compile(r"\b([A-H])\.\s+")

def parse_options_v2(options_str: str) -> List[str]:
    """Parse a Video-MME-v2 'options' string into a list of choice texts.

    Input examples:
        'A. Malaysian. B. British. C. Singaporean. D. German. E. Canadian. ...'
        'A. 4. B. 1. C. 7. D. 2. E. 0. F. 5. G. 6. H. 3.'

    Returns a list of strings, in order A, B, C, ... up to whatever is present.
    """
    s = str(options_str).strip()
    if not s:
        return []
    # Split on " A. " / " B. " etc.; keep the leading "A. " too.
    # We split such that each segment is a full chunk for one letter.
    # Trick: prepend a marker so the first split yields meaningful pieces.
    parts = _OPT_SPLIT_RE.split(" " + s)
    # parts looks like: ['', 'A', 'Malaysian.', 'B', 'British.', ...]
    items = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i].strip()
        content = parts[i + 1].strip()
        # Strip a trailing "." (some entries end with ".")
        if content.endswith("."):
            content = content[:-1].rstrip()
        items[letter] = content
    return [items[L] for L in LETTERS_V2 if L in items]


# ─────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────

PROMPT_V2_NO_SUB = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.\n"
    "Question: {question}\n"
    "{options}"
)

PROMPT_V2_WITH_SUB = (
    "This video's subtitles are listed below:\n"
    "{subtitle}\n\n"
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.\n"
    "Question: {question}\n"
    "{options}"
)


def build_prompt_v2(question: str, options: List[str],
                     subtitle: Optional[str] = None) -> str:
    """Build the V2 MCQ prompt. Pass subtitle=None to use the no-subtitle variant."""
    valid = [o for o in (options or []) if o and str(o).strip()]
    if not valid:
        # Fall back to a generic prompt if option parsing failed
        return f"Question: {question}\nAnswer briefly."
    opts_text = "\n".join(f"{LETTERS_V2[i]}. {o}." for i, o in enumerate(valid)
                          if i < len(LETTERS_V2))
    if subtitle:
        return PROMPT_V2_WITH_SUB.format(question=question, options=opts_text,
                                         subtitle=subtitle.strip())
    return PROMPT_V2_NO_SUB.format(question=question, options=opts_text)


# ─────────────────────────────────────────────────────────────────────
# Answer checking (A-H)
# ─────────────────────────────────────────────────────────────────────

_LETTER_RE_HEAD = re.compile(r"^\(?([A-H])\)?[.\s]*", re.IGNORECASE)
_LETTER_RE_ANY  = re.compile(r"(?<![a-zA-Z])([A-H])(?![a-zA-Z])")
_BOXED_RE       = re.compile(r"\\boxed\{([A-Ha-h])\}")

def check_answer_v2(predicted: str, correct: str,
                     options: Optional[List[str]] = None) -> bool:
    """Lenient MCQ answer extraction supporting A–H. Same shape as v1's check_answer."""
    correct = str(correct).strip().upper()
    predicted = str(predicted).strip()

    if len(correct) != 1 or correct not in "ABCDEFGH":
        return predicted.upper() == correct

    # 1. \boxed{X}
    boxed = _BOXED_RE.findall(predicted)
    if boxed:
        return boxed[-1].upper() == correct

    # 2. Strip "Answer:" / "Final Answer:" prefixes
    cleaned = re.sub(r"^(?:final\s*)?answer\s*[:is]*\s*", "", predicted,
                     flags=re.IGNORECASE).strip()

    # 3. Standalone letter at start: "(C)", "C.", "C "
    m = _LETTER_RE_HEAD.match(cleaned)
    if m:
        return m.group(1).upper() == correct

    # 4. First isolated A-H letter
    m = _LETTER_RE_ANY.search(cleaned.upper())
    if m:
        return m.group(1) == correct

    return False


# ─────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────

def _find_video_v2(videos_root: Path, video_id: str) -> Optional[str]:
    """Locate the .mp4 file for a given video_id under common subdirs."""
    for sub in ["videos_unzipped", "videos", "data", "."]:
        d = videos_root / sub
        if not d.is_dir():
            continue
        for ext in [".mp4", ".mkv", ".webm", ".avi"]:
            cand = d / f"{video_id}{ext}"
            if cand.exists():
                return str(cand)
    # Fallback: recursive search (slow; only used if direct lookup fails)
    for ext in [".mp4", ".mkv", ".webm"]:
        for hit in videos_root.rglob(f"{video_id}{ext}"):
            return str(hit)
    return None


def _read_subtitle(subtitles_root: Optional[Path], video_id: str,
                    join: bool = True) -> Optional[str]:
    """Read a Video-MME-v2 subtitle JSONL into a single concatenated string."""
    if subtitles_root is None:
        return None
    for ext in [".jsonl", ".txt"]:
        path = subtitles_root / f"{video_id}{ext}"
        if path.exists():
            try:
                import json
                lines = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            lines.append(obj.get("text", ""))
                        except Exception:
                            lines.append(line)
                return " ".join(x for x in lines if x).strip() if join else "\n".join(lines)
            except Exception:
                return None
    return None


def load_video_mme_v2(data_dir: str,
                       videos_subdir: Optional[str] = None,
                       subtitles_subdir: Optional[str] = None,
                       with_subtitle: bool = False):
    """Load Video-MME-v2 samples for our eval pipeline.

    Each returned dict has:
        video_path, video_id, question, question_id, options (list[str]),
        answer (single uppercase letter A-H),
        level (str),
        group_type ('logic' | 'relevance'),
        group_structure (str),
        second_head, third_head,
        subtitle (str | None) — only populated if with_subtitle=True
        # compatibility shims with our existing eval code:
        duration_bin = level    (so per-bin breakdowns just work)
        id           = question_id
    """
    import pandas as pd

    root = Path(data_dir)
    parquet = root / "test.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"No test.parquet at {parquet}")

    df = pd.read_parquet(parquet)

    subs_root = None
    if with_subtitle:
        subs_root = root / (subtitles_subdir or "subtitles")
        if not subs_root.is_dir():
            subs_root = None  # silently skip subtitles if folder missing

    samples = []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        video_path = _find_video_v2(root if videos_subdir is None else (root / videos_subdir),
                                    vid)
        if video_path is None:
            # Try the parent dir (for the case where videos_subdir == "")
            video_path = _find_video_v2(root, vid)
        if video_path is None:
            continue

        opts = parse_options_v2(row["options"])
        if not opts:
            continue

        sub_text = _read_subtitle(subs_root, vid) if with_subtitle else None

        level_raw = row.get("level")
        level = str(level_raw) if level_raw is not None and str(level_raw) != "nan" else "0"

        samples.append({
            "video_path": video_path,
            "video_id":   vid,
            "question":   str(row["question"]),
            "question_id": str(row["question_id"]),
            "options":    opts,
            "answer":     str(row["answer"]).strip().upper(),
            "level":      level,
            "group_type": str(row["group_type"]),
            "group_structure": str(row["group_structure"]),
            "second_head": str(row.get("second_head") or ""),
            "third_head":  str(row.get("third_head") or ""),
            "subtitle":   sub_text,
            # compat shims
            "duration_bin": level,
            "id":           str(row["question_id"]),
        })

    return samples


# Convenience: bin labels for v2 are the level strings.
BINS_V2 = ["1", "2", "3"]
