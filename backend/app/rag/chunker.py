from typing import Literal

from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
)


# TODO: Make this a factory instead
def get_chunker(
    chunking_type: Literal["semantic", "recursive", "md"],
) -> TextSplitter:
    if chunking_type == "semantic":
        return SentenceTransformersTokenTextSplitter()
    if chunking_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=1021,
            chunk_overlap=10,
        )
    if chunking_type == "md":
        return MarkdownTextSplitter()
    return RecursiveCharacterTextSplitter()
