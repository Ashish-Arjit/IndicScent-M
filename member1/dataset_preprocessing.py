"""
member1/dataset_preprocessing.py
Responsible for dataset cleaning, label standardization, missing-value handling,
synthetic sample generation (for instant reproducibility), and stratified splitting.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_MEMES_DIR,
    LABEL_MAP,
    SPLIT_RATIOS,
    RANDOM_SEED,
    ensure_directories
)


def standardize_label(raw_label: Any) -> Optional[str]:
    """
    Map various raw dataset label variants (e.g., from Memotion or IndicSentiment)
    to standard 3 classes: 'positive', 'neutral', 'negative'.
    """
    if pd.isna(raw_label):
        return None
    val = str(raw_label).strip().lower()

    if val in ["positive", "pos", "very_positive", "2", "happy"]:
        return "positive"
    elif val in ["negative", "neg", "very_negative", "0", "sad", "angry"]:
        return "negative"
    elif val in ["neutral", "neu", "1"]:
        return "neutral"
    return None


def generate_sample_meme_image(
    output_path: Path,
    caption: str,
    sentiment: str,
    bg_color: Tuple[int, int, int] = (240, 240, 245)
) -> None:
    """Generate a realistic synthetic meme image with text overlay for offline benchmarking."""
    img = Image.new("RGB", (300, 300), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Accent color based on sentiment
    accent_map = {
        "positive": (34, 197, 94),   # Emerald green
        "neutral": (100, 116, 139),  # Slate
        "negative": (239, 68, 68)    # Crimson red
    }
    accent = accent_map.get(sentiment, (59, 130, 246))

    # Draw header banner
    draw.rectangle([(0, 0), (300, 45)], fill=accent)
    draw.text((15, 12), f"MEME #{output_path.stem.upper()}", fill=(255, 255, 255))

    # Draw center box simulating visual meme content
    draw.rectangle([(30, 65), (270, 205)], fill=(255, 255, 255), outline=accent, width=2)
    sentiment_emoji = {"positive": "(^_^)", "neutral": "(-_-)", "negative": "(>_<)"}
    draw.text((125, 125), sentiment_emoji.get(sentiment, "[IMAGE]"), fill=accent)

    # Wrap caption across bottom
    words = caption.split()
    line1 = " ".join(words[:5])
    line2 = " ".join(words[5:10]) if len(words) > 5 else ""
    draw.text((20, 225), line1, fill=(30, 41, 59))
    if line2:
        draw.text((20, 250), line2, fill=(30, 41, 59))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=90)


def create_sample_benchmark_dataset(num_samples: int = 60) -> pd.DataFrame:
    """
    Create a balanced, realistic sample dataset combining English, Hindi,
    and Hinglish memes with paired images. Enables instant pipeline testing.
    """
    ensure_directories()
    
    sample_texts = [
        # Positive (English & Hinglish)
        ("Passed the semester exam without any backlogs! Party toh banti hai bhai!", "positive"),
        ("Weekend plans got approved by mom, feeling like a boss today!", "positive"),
        ("Finally code compiled on the first attempt without warnings!", "positive"),
        ("Diwali bonus credited in account, zindagi ban gayi!", "positive"),
        ("College placement drive cleared, very grateful and excited for new journey!", "positive"),
        ("Biryani order arrived 15 minutes before estimated time, best feeling!", "positive"),
        ("Chai and rains in Delhi, pure bliss and peace.", "positive"),
        ("Team won the hackathon grand prize after 36 hours of coding!", "positive"),
        ("Mummy made my favorite gulab jamun today, absolute perfection!", "positive"),
        ("Project demo was a massive success, professor gave full marks!", "positive"),
        
        # Neutral (English & Hinglish)
        ("Today temperature in campus is 32 degrees celsius.", "neutral"),
        ("Class rescheduled from 10 AM to 2 PM as per official notice.", "neutral"),
        ("Please submit lab assignment notebook before Thursday 5 PM.", "neutral"),
        ("Metro train frequency reduced on yellow line for track maintenance.", "neutral"),
        ("Waiting at the bus stop for route 412 since twenty minutes.", "neutral"),
        ("Syllabus for midterm exams has been updated on university portal.", "neutral"),
        ("Lecture hall 4 projector is currently undergoing routine maintenance.", "neutral"),
        ("The attendance threshold is seventy-five percent for semester exam.", "neutral"),
        ("Library will remain open till 9 PM during examination week.", "neutral"),
        ("Results will be declared next Monday afternoon via ERP portal.", "neutral"),
        
        # Negative (English & Hinglish)
        ("WiFi disconnected right in the middle of online coding test!", "negative"),
        ("8 AM morning class on a freezing winter Monday is pure torture.", "negative"),
        ("Train delayed by five hours and cafeteria food is completely cold.", "negative"),
        ("Laptop motherboard crashed right before the final project submission!", "negative"),
        ("Auto driver asked three hundred rupees for two kilometer ride, loot macha rakhi hai.", "negative"),
        ("Electricity cut in hostel during 42 degree summer afternoon.", "negative"),
        ("Lost my college ID card on examination day, huge hassle.", "negative"),
        ("Worst customer support ever, waiting on call since 45 minutes.", "negative"),
        ("Spilled hot coffee on my newly printed project report file.", "negative"),
        ("Traffic jam on highway caused me to miss the flight, total disaster.", "negative"),

        # Indian-Language Code-Mixed / Sarcastic Cases (Multimodal Disagreement Candidates)
        ("Oh wonderful, another meeting that could have easily been an email.", "negative"),
        ("Kya baat hai, exam se ek raat pehle syllabus hi change ho gaya!", "negative"),
        ("Great job guys, broken production code pushed straight to master on Friday!", "negative"),
        ("Super happy to see petrol price touching new all-time record high today!", "negative"),
        ("Bohot badiya, hostel mess me phir se aalu gobhi banayi hai.", "negative"),
        ("Wah kya scene hai, salary aayi nahi ki rent aur EMI ne uda di!", "negative"),
        ("Yes of course, studying 14 units in 3 hours is totally manageable.", "negative"),
        ("Subah 6 baje alarm bajna is the most pleasant experience ever.", "negative"),
        ("Congratulations to me for locking myself outside hostel room at midnight.", "negative"),
        ("Brilliant! Just when I was about to save, power went off completely.", "negative"),

        # Extra Positive Hindi/Indic
        ("Bhai ka placement ho gaya Google mein! Bohot khushi ka din hai!", "positive"),
        ("Aaj mausam bohot suhana hai, chai pakode ke sath maze kar rahe hain.", "positive"),
        ("Professor ne presentation ki bohot tareef ki, sab khush hain!", "positive"),
        ("Hard work paid off, semester topper award mila aaj!", "positive"),
        ("Family ke sath Goa trip confirm ho gayi, super excited!", "positive"),
        ("Dost ne birthday pe surprise party organize ki, speechless ho gaya!", "positive"),
        ("Internship stipend credit ho gaya, aaj sabko treat meri taraf se!", "positive"),
        ("Team mates helped debug the error in 5 minutes, best group ever!", "positive"),
        ("Indian cricket team won thrilling match on the last ball!", "positive"),
        ("Ghar ka khana after six months of hostel food, feeling heavenly!", "positive"),

        # Extra Neutral Hindi/Indic
        ("Aaj college ki bus 15 minute late aayegi.", "neutral"),
        ("Semester fee submit karne ki last date 25th hai.", "neutral"),
        ("Canteen ka menu Monday se change hoga notice board pe likha hai.", "neutral"),
        ("Seminar room 302 me presentation schedule ki gayi hai.", "neutral"),
        ("Practical exam ke groups roll number ke hisaab se banaye gaye hain.", "neutral"),
        ("Assignment questions portal pe upload kar diye gaye hain.", "neutral"),
        ("Exam hall ticket download link email par send kiya gaya hai.", "neutral"),
        ("Sports day registration desk auditorium ke paas khula hai.", "neutral"),
        ("ID card renewal ke liye passport photo jama karni padegi.", "neutral"),
        ("Next semester registration form kal subah 10 baje open hoga.", "neutral")
    ]

    records = []
    for i, (text, sentiment) in enumerate(sample_texts[:num_samples]):
        sample_id = f"MEME_{i+1:04d}"
        img_name = f"{sample_id}.jpg"
        img_path = SAMPLE_MEMES_DIR / img_name
        
        # Invert or align image visual theme to simulate realistic meme tension
        # E.g. sarcastic memes might have a positive/neutral banner with negative text
        generate_sample_meme_image(img_path, text, sentiment)

        records.append({
            "sample_id": sample_id,
            "text": text,
            "image_path": str(img_path.resolve()),
            "raw_sentiment": sentiment,
            "sentiment": sentiment,
            "label": LABEL_MAP[sentiment]
        })

    df = pd.DataFrame(records)
    raw_csv = RAW_DATA_DIR / "sample_memes.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Generated {len(df)} sample paired meme records to {raw_csv}")
    return df


def clean_and_split_dataset(
    df: pd.DataFrame,
    test_size: float = SPLIT_RATIOS["test"],
    val_size: float = SPLIT_RATIOS["val"],
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean dataset:
    - Standardize sentiment labels
    - Remove missing texts, missing images, and duplicates
    - Create stratified train, validation, and test splits
    """
    ensure_directories()
    df_clean = df.copy()

    # Standardize label
    df_clean["sentiment"] = df_clean["sentiment"].apply(standardize_label)
    df_clean = df_clean.dropna(subset=["sentiment", "text"])

    # Ensure label int column
    df_clean["label"] = df_clean["sentiment"].map(LABEL_MAP)

    # Drop duplicates based on text
    df_clean = df_clean.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Filter out missing images if image_path exists
    if "image_path" in df_clean.columns:
        df_clean = df_clean[df_clean["image_path"].apply(lambda p: Path(str(p)).exists())].reset_index(drop=True)

    # Stratified Split: First split off test set
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_seed,
        stratify=df_clean["label"]
    )

    # Split train_val into train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_seed,
        stratify=train_val_df["label"]
    )

    # Save to processed directory
    df_clean.to_csv(PROCESSED_DATA_DIR / "all_clean.csv", index=False)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"Dataset Preprocessing Complete:")
    print(f"  Total Clean: {len(df_clean)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df_clean):.1%})")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df_clean):.1%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df_clean):.1%})")

    return train_df, val_df, test_df


def load_or_create_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for Member 1:
    Checks if raw data is provided, otherwise generates a benchmark sample dataset,
    cleans, and returns stratified splits.
    """
    ensure_directories()
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))

    if raw_files:
        print(f"Found raw dataset: {raw_files[0]}")
        df_raw = pd.read_csv(raw_files[0])
    else:
        print("No raw CSV found in data/raw/. Generating reproducible sample benchmark dataset...")
        df_raw = create_sample_benchmark_dataset()

    return clean_and_split_dataset(df_raw)


if __name__ == "__main__":
    load_or_create_dataset()
