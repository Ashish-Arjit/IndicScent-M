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


def clean_text(text: str) -> str:
    """
    Clean text for sentiment analysis:
    - Replace URLs and web links
    - Replace user handles (@mention)
    - Map emoticons and emojis to sentiment markers
    - Reduce character repetitions (e.g., 'soooo good' -> 'soo good')
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
