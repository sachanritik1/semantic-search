# app/services/tokenizer.py

from typing import List
import tiktoken


def tokenize(text: str, model: str = "gpt-4o") -> List[int]:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return tokens

def detokenize(tokens: List[int], model: str = "gpt-4o") -> str:
    encoding = tiktoken.encoding_for_model(model)
    text = encoding.decode(tokens)
    return text