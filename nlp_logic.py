import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = genai.Client()

# Ordered from easiest to hardest
JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1", "N0"]


def generate_json(prompt, schema, max_output_tokens=2048):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
            max_output_tokens=max_output_tokens
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        return json.loads(response.text)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Unable to parse Gemini JSON response: {response.text[:300]}"
        ) from error


def analyze_JLPT(
    text,
    output_csv,
    level="N3",
    max_words=20,
    no_N0_words=True,
    include_example=False
):
    """
    Translate Japanese text to Thai and mine its vocabulary.

    level           : easiest level to keep ("N3" keeps N3 and harder)
    max_words       : maximum number of words
    no_N0_words     : if True, exclude N0 words
    include_example : if True, add the source sentence as a fifth column

    Returns (translation, DataFrame).
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    if level not in JLPT_LEVELS:
        raise ValueError(f"level must be one of {JLPT_LEVELS}.")

    if not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer.")

    if level == "N0" and no_N0_words:
        raise ValueError("level='N0' requires no_N0_words=False.")

    text = text.strip()

    allowed_levels = JLPT_LEVELS[JLPT_LEVELS.index(level):]

    if no_N0_words:
        allowed_levels = [x for x in allowed_levels if x != "N0"]

    schema = {
        "type": "OBJECT",
        "properties": {
            "translation": {"type": "STRING"},
            "vocabulary": {
                "type": "ARRAY",
                "maxItems": max_words,
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "surface": {"type": "STRING"},
                        "expression": {"type": "STRING"},
                        "reading": {"type": "STRING"},
                        "definition": {"type": "STRING"},
                        # All levels stay available, so that the model
                        # is not forced to mislabel an easy word
                        "jlpt": {"type": "STRING", "enum": JLPT_LEVELS}
                    },
                    "required": [
                        "surface",
                        "expression",
                        "reading",
                        "definition",
                        "jlpt"
                    ]
                }
            }
        },
        "required": ["translation", "vocabulary"]
    }

    prompt = f"""
คุณคือผู้เชี่ยวชาญภาษาญี่ปุ่นและศัพท์ระดับสูง และเป็นนักแปลขั้นสูงญี่ปุ่น-ไทย

ข้อความต้นฉบับ:

{text}

งานที่ 1: การแปล (translation)
แปลข้อความทั้งหมดเป็นภาษาไทยอย่างครบถ้วนและสละสลวย
โดยอิงภาษาศาสตร์สังคมปฏิบัติ (sociopragmatics)
รักษาน้ำเสียง ระดับภาษา และการแบ่งย่อหน้าของต้นฉบับ
ห้ามสรุปหรือตัดทอน

งานที่ 2: การเก็บศัพท์ (vocabulary)
เลือกศัพท์ (lexemes) ที่มีคุณค่าในการเรียนรู้ ไม่เกิน {max_words} คำ
เฉพาะระดับ {", ".join(allowed_levels)} เท่านั้น
หากมีไม่ถึงจำนวนดังกล่าว ไม่ต้องเติมให้ครบ

ข้อกำหนด:

1. ไม่เลือกรูปไวยากรณ์ (文法)
2. ไม่เลือกคำช่วยทางไวยากรณ์ (助詞)
3. surface = รูปที่ปรากฏจริงในข้อความ คัดลอกมาตรงตัวอักษร (เช่น 味わい)
4. expression = รูปพจนานุกรม (辞書形) ของคำนั้น (เช่น 味わう)
5. reading = คำอ่านของ expression เป็นฮิรางานะ (คำคาตากานะให้คงเป็นคาตากานะ)
6. definition = ความหมายภาษาไทยตามบริบท
7. jlpt = ระดับความยาก โดยอิงจากผู้เรียนภาษาญี่ปุ่นทั่วไป ประกอบกับเกณฑ์นี้
   - N5 = ศัพท์รูปธรรม กริยาพื้นฐาน จำนวน เวลา
   - N4 = ศัพท์สภาวะทางอารมณ์ เครื่องใช้ภายในบ้าน รูปสุภาพ (敬語)
   - N3 = ศัพท์นามธรรม ศัพท์ภาษาพูด
   - N2 = ศัพท์ธุรกิจ สำนวนทางการ ภาษาจดหมาย สัทพจน์
   - N1 = คันจิควบกล้ำ (四字熟語) ศัพท์วรรณกรรม
   - N0 = ระดับสูงกว่า N1: ศัพท์เฉพาะทาง เช่น คันจิโบราณ (旧字体)
     การสอบคันจิ (เช่น 漢検2級) วรรณกรรมคลาสสิก (เช่น 故事成語)
"""

    result = generate_json(prompt, schema, max_output_tokens=8192)

    translation = str(result.get("translation", "")).strip()
    vocabulary = result.get("vocabulary")

    if not translation:
        raise RuntimeError("Gemini returned no translation.")

    if not isinstance(vocabulary, list):
        raise RuntimeError("Invalid vocabulary response.")

    # Split after Japanese or Western end punctuation, or at line breaks
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[。！？!?])(?![」』])|\n+", text)
        if s.strip()
    ]

    rows = []
    seen_words = set()

    for item in vocabulary:

        if not isinstance(item, dict):
            continue

        surface = str(item.get("surface", "")).strip()
        expression = str(item.get("expression", "")).strip()
        reading = str(item.get("reading", "")).strip()
        definition = str(item.get("definition", "")).strip()
        jlpt = item.get("jlpt")

        if not all([surface, expression, reading, definition]):
            continue

        if jlpt not in allowed_levels:
            continue

        # Japanese has no spaces, so use a plain substring check
        if surface not in text:
            continue

        if expression in seen_words:
            continue

        row = {
            "expression": expression,
            "reading": reading,
            "definition": definition,
            "jlpt": jlpt
        }

        if include_example:
            row["example"] = next(
                (s for s in sentences if surface in s), ""
            )

        rows.append(row)
        seen_words.add(expression)

        if len(rows) == max_words:
            break

    # Easiest to hardest
    rows.sort(key=lambda row: JLPT_LEVELS.index(row["jlpt"]))

    columns = ["expression", "reading", "definition", "jlpt"]

    if include_example:
        columns.append("example")

    df = pd.DataFrame(rows, columns=columns)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Vocabulary CSV saved to: {output_path}")
    print(f"Total valid vocabulary words: {len(df)}")

    return translation, df


if __name__ == "__main__":
    article = """
固い地べたの感触を顔面に味わい、彼は自分がうつ伏せに倒れたのだと気付いた。

全身に力が入らず、手先の感覚はすでにない。

ただ、喉をかきむしりたくなるほどの熱が体の真ん中を支配している。
"""

    translation, df = analyze_JLPT(article, "data/vocabulary.csv")

    print("\nTranslation:")
    print(translation)

    print("\nVocabulary Results:")
    print(df.to_string(index=False))