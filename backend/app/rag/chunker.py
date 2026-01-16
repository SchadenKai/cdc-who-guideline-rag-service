from typing import Literal

from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
)


def get_chunker(
    chunking_type: Literal["semantic", "recursive", "md"],
) -> TextSplitter:
    if chunking_type == "semantic":
        return SentenceTransformersTokenTextSplitter()
    if chunking_type == "recursive":
        return RecursiveCharacterTextSplitter()
    if chunking_type == "md":
        return MarkdownTextSplitter()
    return RecursiveCharacterTextSplitter()
