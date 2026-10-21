# IndicSent-M Data Directory

This directory stores raw and preprocessed datasets for multimodal and Indian-language sentiment analysis.

## Structure

- `raw/`: Stores pristine original datasets (e.g., Memotion 7K CSV & images, IndicSentiment CSVs).
- `processed/`: Stores cleaned metadata, stratified train/val/test splits (`train.csv`, `val.csv`, `test.csv`), and text quality summaries.
- `sample_memes/`: Contains generated or sample meme images paired with captions for immediate offline testing and demonstration without downloading multi-gigabyte archives.

## Supported Experiment Tracks

### Track 1: Paired Memes (Multimodal Benchmark)
- **Primary candidate:** Memotion Dataset 7K (SemEval-2020 Task 8).
- **Format:** Paired image file + extracted text/OCR caption + sentiment label (`positive`, `neutral`, `negative`).
- **Splits:** 70% Train, 15% Validation, 15% Test.

### Track 2: Indian-Language Text (Multilingual Benchmark)
- **Primary candidate:** AI4Bharat IndicSentiment / DravidianCodeMix.
- **Format:** Multilingual Indian text (Hindi, Hinglish, Tamil, Telugu, etc.) + 3-class sentiment label.

## How to Load Your Own Dataset
Place raw CSV files and image folders into `data/raw/` and execute:
```bash
python main.py --prepare-data
```
