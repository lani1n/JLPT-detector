# JLPT Detector (V2)

> **Krittamet Supharatana**

A Japanese-to-Thai translation and vocabulary-mining tool powered by Google's Gemini API.

## Overview

JLPT Detector is a Streamlit web application designed to support Japanese language learning. It provides two main features:

- **Japanese–Thai Translation:** Translates Japanese passages into Thai while aiming to preserve their original meaning, tone, and register.
- **Vocabulary Extraction:** Identifies useful Japanese vocabulary, normalizes words into their dictionary forms, and assigns estimated JLPT difficulty levels.

The project is intended particularly for intermediate-to-advanced Japanese learners who want to explore vocabulary beyond basic materials.

## V2 Highlights

- Gemini-powered Japanese–Thai translation and vocabulary extraction.
- Dictionary-form normalization and contextual reading identification.
- JLPT classification based on an unofficial vocabulary dataset.
- A custom **N0** category for selected technical terminology, rare orthographic forms, and hyper-advanced lexemes beyond the conventional JLPT examination.
- Adjustable JLPT difficulty threshold and vocabulary extraction limit.
- Tabular vocabulary results with .CSV export.

> **Note:** JLPT levels are unofficial estimates based on the selected vocabulary dataset and supplementary classification rules. N0 is a custom category, not an official JLPT level.

## Background

I am an advanced Japanese learner. This project combines language learning with NLP, particularly vocabulary extraction and lexical classification.

Rather than simply translating a passage or lexical mapping, I wanted to build a tool that help learners examine the vocabulary contained within it.

## Vocabulary Dataset

**Dataset:** [JLPT_Vocabulary](https://github.com/Bluskyo/JLPT_Vocabulary)

- **Original source:** Vocabulary data from [Tanos](https://www.tanos.co.uk/jlpt/), licensed under Creative Commons by Jonathan Waller.
- **Dataset preparation:** Rearranged and normalized by Isak Mikalsen.
- **Format:** JSON.
- **Coverage:** Approximately 8,138 written expressions.
- **Classification:** N1–N5.

The dataset distinguishes written expressions, meaning that different orthographic forms such as 挨拶 and あいさつ are represented separately. This is a useful distinction for vocabulary learning, although it means that the dataset does not include every possible spelling.

I chose this dataset because it provides a relatively clean and practical reference for JLPT vocabulary classification. Since official JLPT vocabulary lists are no longer available, this project thus uses an unofficial dataset.

### Dataset Citation

```bibtex
@misc{Bluskyo_JLPT_Vocabulary,
    author       = {Mikalsen, Isak},
    title        = {{JLPT\_Vocabulary}},
    year         = {2026},
    month        = apr,
    version      = {1.5-2026-04-30},
    howpublished = {\url{https://github.com/Bluskyo/JLPT_Vocabulary}},
    note         = {Data set. Accessed: 2026-11-28}
}
```

## Development Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management. See `pyproject.toml` for the project dependencies.

I recommend `uv` as a lightweight alternative to heavier environment-management workflows. This project does not use `anaconda`.

## Planned Improvements

- Improved Thai typography and font rendering
- More readable interface elements
- Better responsive layout
- Further refinement of Japanese lexical analysis