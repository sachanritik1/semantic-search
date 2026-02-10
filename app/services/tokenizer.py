# app/services/tokenizer.py

from typing import List
import tiktoken


def get_tokens(text: str, model: str = "gpt-4o") -> List[int]:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return tokens
