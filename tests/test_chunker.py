# tests/test_chunker.py

import pytest
from app.services.chunker import sliding_window_chunk
from app.services.tokenizer import tokenize, detokenize

def test_sliding_window_overlap():
    text = "A B C D E F G H I"

    chunks = sliding_window_chunk(
        text,
        chunk_size=5,
        overlap=2,
        tokenize=tokenize,
        detokenize=detokenize,
    )

    assert chunks == [
        "A B C D E",
        "D E F G H",
        "G H I",
    ]


def test_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        sliding_window_chunk(
            "A B C",
            chunk_size=3,
            overlap=3,
            tokenize=tokenize,
            detokenize=detokenize,
        )
