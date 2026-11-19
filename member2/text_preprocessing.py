"""
member2/text_preprocessing.py
Preprocesses multilingual, code-mixed (Hinglish/Indian languages), and social media meme text.
Cleans noise, normalizes characters, and preserves sentiment-bearing emojis and markers.
"""

import re
import sys
from pathlib import Path
from typing import List, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Simple dictionary mapping common social media emojis / emoticons to sentiment tokens
EMOJI_SENTIMENT_MAP = {
    ":)": " emoji_positive ",
    ":-)": " emoji_positive ",
    ":D": " emoji_positive ",
    "😊": " emoji_positive ",
    "😃": " emoji_positive ",
    "🔥": " emoji_positive ",
    "❤️": " emoji_positive ",
    "👍": " emoji_positive ",
    "😂": " emoji_positive ",
    "🤣": " emoji_positive ",
    ":(": " emoji_negative ",
    ":-(": " emoji_negative ",
    "😢": " emoji_negative ",
    "😭": " emoji_negative ",
    "😡": " emoji_negative ",
    "💔": " emoji_negative ",
    "👎": " emoji_negative ",
    ":/": " emoji_neutral ",
    ":-|": " emoji_neutral ",
    "😐": " emoji_neutral ",
    "🤔": " emoji_neutral "
}

# Task 2.2: Expanded Hinglish and Romanized Indic Phonetic Normalization Map
# Unifies phonetic spelling variants common in Indian social media memes and chats
HINGLISH_PHONETIC_MAP = {
    "bohot": "bahut",
    "boht": "bahut",
    "bht": "bahut",
    "bhut": "bahut",
    "bahot": "bahut",
    "bhout": "bahut",
    "khus": "khush",
    "khusi": "khush",
    "khushi": "khush",
    "accha": "achha",
    "acha": "achha",
    "achaa": "achha",
    "bhayi": "bhai",
    "bhaiya": "bhai",
    "bhaii": "bhai",
    "nhi": "nahi",
    "nahin": "nahi",
    "nhn": "nahi",
    "nai": "nahi",
    "aj": "aaj",
    "kl": "kal",
    "paise": "paisa",
    "paissa": "paisa",
    "jindagi": "zindagi",
    "zindgi": "zindagi",
    "ekdm": "ekdam",
    "shaandar": "shandar",
    "badiya": "badhiya",
    "bdhiya": "badhiya",
    "kia": "kya",
    "kyu": "kyun",
    "yar": "yaar",
    "yr": "yaar",
    "shi": "sahi",
    "mst": "mast",
    "bekaar": "bekar",
    "gya": "gaya",
    "gyi": "gayi",
    "gye": "gaye",
    "rha": "raha",
    "rhi": "rahi",
    "rhe": "rahe",
    "hwa": "hua",
    "hwi": "hui",
    "hue": "huye",
    "lkn": "lekin",
    "mgr": "magar",
    "sbse": "sabse",
    "chaiye": "chahiye",
    "chahie": "chahiye",
    "thik": "theek",
    "thk": "theek",
    "shukria": "shukriya",
    "dst": "dost",
    "maja": "maza",
    "mza": "maza",
    "tarif": "tareef"
}


def normalize_hinglish(text: str) -> str:
    """
    Task 2.2: Transliteration & Romanized Hinglish phonetic normalizer.
    Replaces noisy phonetic spelling variations with canonical vocabulary tokens.
    Handles capitalization and punctuation transparently.
    """
    words = text.split()
    normalized = []
    for w in words:
        w_clean = re.sub(r"^[^\w]+|[^\w]+$", "", w.lower())
        mapped = HINGLISH_PHONETIC_MAP.get(w_clean, w.lower())
        normalized.append(mapped)
    return " ".join(normalized)


def clean_text(text: str) -> str:
    """
    Clean text for sentiment analysis:
    - Replace URLs and web links
    - Replace user handles (@mention)
    - Map emoticons and emojis to sentiment markers
    - Reduce character repetitions (e.g., 'soooo good' -> 'soo good')
    - Normalize Romanized Hinglish spelling variants
    - Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert URLs
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)

    # Convert user mentions
    text = re.sub(r"@\w+", " user_mention ", text)

    # Convert emoticons / emojis
    for emo, tag in EMOJI_SENTIMENT_MAP.items():
        text = text.replace(emo, tag)

    # Remove generic non-word noise but retain Devanagari / Indic Unicode scripts & Latin chars
    # Unicode ranges: \u0900-\u097F is Devanagari (Hindi, Marathi, Nepali, Sanskrit)
    # \u0980-\u09FF is Bengali, \u0B80-\u0BFF is Tamil, \u0C00-\u0C7F is Telugu
    text = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F]", " ", text)

    # Compress multiple repeated characters (> 2 repetitions to 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Lowercase Latin characters
    text = text.lower()

    # Apply Hinglish phonetic normalization
    text = normalize_hinglish(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_corpus(texts: Union[List[str], "pd.Series"]) -> List[str]:
    """Batch preprocessing of a list or pandas Series of texts."""
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    sample = "Exam postponed!! Soooo happy :D checkout https://portal.edu @student_council क्या बात है!!"
    print("Original Text:", sample)
    print("Cleaned Text: ", clean_text(sample))
