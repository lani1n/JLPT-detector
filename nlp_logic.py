import os
import re
import json
from pathlib import Path
from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
from jlpt_lookup import JLPT_LEVELS, allowed_levels, classify, load_data


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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

    if not isinstance(max_words, int) or max_words < 1:
        raise ValueError("max_words must be a positive integer.")

    # Also validates level
    levels = allowed_levels(level, include_n0=not no_N0_words)

    # Fail before calling Gemini if a data file is missing
    load_data()

    text = text.strip()

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
# บทบาท: คุณคือผู้เชี่ยวชาญภาษาญี่ปุ่นและศัพท์ระดับสูง และเป็นนักแปลขั้นสูงญี่ปุ่น-ไทย

ข้อความต้นฉบับ:

{text}

# หน้าที่
## งานที่ 1: การแปล (translation)
จงแปลข้อความทั้งหมดเป็นภาษาไทยอย่างครบถ้วนและสละสลวย
โดยอิงภาษาศาสตร์สังคมปฏิบัติ (sociopragmatics)
รักษาน้ำเสียง (tone) ระดับภาษา (register) และการแบ่งย่อหน้าของต้นฉบับ
ห้ามสรุปหรือตัดทอน

## งานที่ 2: การเก็บคำศัพท์ (vocabulary)
เลือกศัพท์ (lexemes) ที่มีคุณค่าในการเรียนรู้ ไม่เกิน {max_words} คำ
เฉพาะระดับ {", ".join(levels)} เท่านั้น
หากมีไม่ถึงจำนวนดังกล่าว ไม่ต้องเติมให้ครบ

# ข้อกำหนด
1. ไม่เลือกรูปไวยากรณ์ (文法)
2. ไม่เลือกคำช่วยทางไวยากรณ์ (助詞)
3. surface = รูปที่ปรากฏจริงในข้อความ คัดลอกมาตรงตัวอักษร (เช่น 味わい、食べさせて)
4. expression = รูปพจนานุกรม (辞書形) ของคำนั้น (เช่น 味わう、食べる)
   ต้องคงรูปเขียนตามต้นฉบับ ห้ามเปลี่ยนเป็นคันจิตัวอื่นหรือรูปเขียนที่พบบ่อยกว่า
   (เช่น 曰う ต้องคงเป็น 曰う ห้ามเปลี่ยนเป็น 言う และคำที่เขียนด้วยคานะให้คงเป็นคานะ)
5. reading = คำอ่านของ expression เป็นอักษรฮิรางานะ (คำคาตากานะให้คงเป็นคาตากานะ)
6. definition = ความหมายภาษาไทยโดยอิงตามบริบททางภาษา (linguistic context)
7. jlpt = ระดับความยาก โดยอิงจากผู้เรียนภาษาญี่ปุ่นทั่วไป ประกอบกับเกณฑ์นี้
   - N5 = ศัพท์รูปธรรม กริยาพื้นฐาน จำนวน เวลา
   - N4 = ศัพท์สภาวะทางอารมณ์ เครื่องใช้ภายในบ้าน รูปสุภาพ (敬語)
   - N3 = ศัพท์นามธรรม ศัพท์ภาษาพูด
   - N2 = ศัพท์ธุรกิจ สำนวนทางการ ภาษาจดหมาย สัทพจน์
   - N1 = คันจิควบกล้ำ (四字熟語) ศัพท์วรรณกรรม
   - N0 = ระดับสูงกว่า N1: ศัพท์เฉพาะทาง เช่น คันจิโบราณ (旧字体)
     การสอบคันจิระดับสูง (เช่น 漢検準1級 หรือสูงกว่า) วรรณกรรมคลาสสิก (เช่น 故事成語)
     รวมถึงรูปเขียนที่ปรากฏน้อย แม้ความหมายจะไม่ยาก เช่น 同訓異字 (曰う、謂う)

# ข้อจำกัดและข้อควรระวัง
หากข้อความต้นฉบับมี
1. ฟูริกานะ (振り仮名) จงเพิกเฉยและห้ามแปลฟูริกานะซ้ำ เช่น 何故なぜ "ทำไม"
2. อักษรพ้องรูป (homographs) จงปฎิบัติตามหลักเกณฑ์นี้อย่างเคร่งครัด
    3.1 หากความหมายต่างกัน จงพิจารณาความหมายที่เหมาะสมกับบริบท
    และเลือกศัพท์นั้นสำหรับงานที่ 2 เช่น 人気 อ่านได้ทั้ง にんき หรือ ひとけ
    3.2 หากความหมายเหมือนกัน จงพิจารณาระดับภาษาของข้อความและประเภทงานเขียน
    เช่น 今日 ควรมีเสียงอ่านใด きょう、こんにち หรือ こんち จงพิจารณาจากข้อความ
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

        if not all([surface, expression, reading, definition]):
            continue

        # List first, then the N0 rule, then Gemini's estimate.
        # This must happen before the level filter.
        jlpt, level_source = classify(expression, reading, item.get("jlpt"))

        if jlpt not in levels:
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
            "jlpt": jlpt,
            "level_source": level_source
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