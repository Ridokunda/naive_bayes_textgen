import re

def preprocess_text(text):
    """
    Cleans raw text: lowercasing, removing HTML tags, punctuation.
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)   # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text) # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    """
    Splits text into list of words.
    """
    return text.split()
