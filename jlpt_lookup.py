import json
import re
from functools import lru_cache
from pathlib import Path


# Ordered from easiest to hardest. N0 is unofficial.
JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1", "N0"]

# Relative to this file, so it works from any working directory
JLPT_PATH = Path(__file__).parent / "dataset" / "jlpt.json"
JOYO_PATH = Path(__file__).parent / "dataset" / "joyo_kanji.txt"

# CJK ideographs. Kana, 々 and punctuation are not matched.
KANJI = re.compile(
    "[㐀-䶿一-鿿豈-﫿\U00020000-\U0003ffff]"
)


@lru_cache(maxsize=1)
def load_jlpt_db():
    """Load the JSON once and cache it for subsequent lookups."""
    if not JLPT_PATH.is_file():
        raise FileNotFoundError(f"JLPT dataset not found: {JLPT_PATH}")

    with JLPT_PATH.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("jlpt.json must contain a JSON object at its root.")

    return data


@lru_cache(maxsize=1)
def load_joyo_kanji():
    """Load the set of 常用漢字 once."""
    if not JOYO_PATH.is_file():
        raise FileNotFoundError(f"Joyo kanji file not found: {JOYO_PATH}")

    lines = JOYO_PATH.read_text(encoding="utf-8-sig").splitlines()

    return frozenset(
        "".join(line for line in lines if not line.startswith("#"))
    )


def load_data():
    """Load both files now, to fail before calling Gemini if one is missing."""
    load_jlpt_db()
    load_joyo_kanji()


def lookup_jlpt(expression, reading):
    """
    Return the listed level of (dictionary form, reading), or None.

    Both must match, so homographs are kept apart (銅: どう / あかがね).
    A pair listed at two levels takes the easier one (作る: N5 and N2).
    """
    levels = [
        item["level"]
        for item in load_jlpt_db().get(expression, [])
        if item.get("reading") == reading
        and item.get("level") in (1, 2, 3, 4, 5)
    ]

    return f"N{max(levels)}" if levels else None


def has_hyogai_kanji(expression):
    """True if the spelling uses any kanji outside 常用漢字 (表外字)."""
    joyo = load_joyo_kanji()

    return any(char not in joyo for char in KANJI.findall(expression))


def classify(expression, reading, estimate):
    """
    Return (level, source). Unofficial N0 needs positive evidence.

    1. list     : the pair is in the JLPT list (never overridden)
    2. rule     : unlisted and written with a 表外字 -> N0
    3. estimate : otherwise, the caller's estimate (Gemini's judgement)
    """
    listed_level = lookup_jlpt(expression, reading)

    if listed_level:
        return listed_level, "list"

    if has_hyogai_kanji(expression):
        return "N0", "rule"

    return estimate, "estimate"


def allowed_levels(minimum_level, include_n0=False):
    """
    Return the levels to keep: minimum_level and harder.

    Example: "N3" gives ["N3", "N2", "N1"], plus "N0" if include_n0.
    """
    if minimum_level not in JLPT_LEVELS:
        raise ValueError(f"level must be one of {JLPT_LEVELS}.")

    if minimum_level == "N0" and not include_n0:
        raise ValueError("level='N0' requires no_N0_words=False.")

    levels = JLPT_LEVELS[JLPT_LEVELS.index(minimum_level):]

    if not include_n0:
        levels = [x for x in levels if x != "N0"]

    return levels


if __name__ == "__main__":
    # Quick check without Gemini: python jlpt_lookup.py
    examples = [
        ("銅", "あかがね"),
        ("銅", "どう"),
        ("人気", "にんき"),
        ("人気", "ひとけ"),
        ("明白", "あからさま"),
        ("作る", "つくる"),
        ("曰う", "いう"),
    ]

    for expression, reading in examples:
        level, source = classify(expression, reading, "(no estimate)")
        print(f"{expression} ({reading}): {level} [{source}]")

    print("\nN3 and above:", allowed_levels("N3"))
    print("N3 and above, with N0:", allowed_levels("N3", include_n0=True))