
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Callable


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,  add_start_index=True,
)

def sliding_window_chunk(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    tokenize: Callable[[str], List[int]],
    detokenize: Callable[[List[int]], str],
) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text: input text
        chunk_size: number of tokens per chunk
        overlap: number of overlapping tokens between chunks
        tokenize: function to convert text -> tokens
        detokenize: function to convert tokens -> text

    Returns:
        List of text chunks
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenize(text)
    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(detokenize(chunk_tokens).strip()) 
        start += step

    return chunks 