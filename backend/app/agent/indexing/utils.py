import hashlib
import re


def clean_chunk_content(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hash_text(text: str) -> str:
    text_bytes = text.encode()
    return hashlib.sha256(text_bytes).hexdigest()
