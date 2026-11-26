# JLPT Detector (V1)
> Krittamet Supharatana

## V1 Summary
- A Streamlit app that (1) translates Japanese to Thai and (2) extracts vocabulary with estimated JLPT levels, using the Gemini API
- JLPT levels are AI estimates. "N0" is a custom level above N1 for technical terms or archaic lexmes

## Further implementation
### Trying to using the dataset for JLPT tagging
- The dataset: Bluskyo JMDict_Extended
- Scope & size: Full JMdict + JLPT tags + pitch + furigana
- Format: JSON
- Reason to use: Pre-joining JMdict entries with cleaned JLPT levels, thus no need to write custom dictionary-matching scripts. JLPT Official is no longer available.
- Citation:

Bluskyo, I. (2026). JMDict_Extended (Version 1.4.1-auto-release-2026-11-24) [Data set]. GitHub. Retrieved November 26, 2026, from https://github.com/Bluskyo/JMDict_Extended

```
@misc{Bluskyo_JMDict_Extended_2026,
    author       = {Bluskyo, Isak},
    title        = {{JMDict\_Extended}},
    year         = {2026},
    month        = nov,
    version      = {1.4.1-auto-release-2026-11-24},
    howpublished = {\url{https://github.com/Bluskyo/JMDict_Extended}},
    note         = {Data set. Accessed: 2026-11-26}
}
```

### UI Improvement
- Thai fonts
- bigger size
- \(optional\) responsive design for mobiles 

### QoL/Bugs with Japanese
- furigana (振り仮名)
- gikun (義訓)
- Chinese characters
- punctuations/regex
- polysemy/homographs

## Nice to have features

### Sentence mining for Anki