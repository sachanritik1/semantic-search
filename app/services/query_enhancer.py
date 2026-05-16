# app/services/query_enhancer.py

from app.services.llm_service import LLMService


class QueryEnhancer:
    """Simple LLM-based query enhancer.

    It uses the application's LLMService to rewrite/expand user queries to
    improve retrieval quality. The enhancer uses a fixed prompt and returns
    the enhancer's text output as the enhanced query.
    """

    def __init__(self, llm_service: LLMService, enhancer_model: str | None = None):
        self.llm_service = llm_service
        self.enhancer_model = enhancer_model

    def enhance(self, query: str) -> str:
        prompt = (
            "You are a helpful assistant that rewrites user search queries to be "
            "clearer, more specific, and more likely to retrieve relevant documents. "
            "Preserve the original intent and add clarifying details or synonyms as needed.\n\n"
            f"Original query: \"{query}\"\n\n"
        
            "Return the enhanced query only, no other text, symbol or anything else."
        )
        

        response = self.llm_service.generate_text(
            prompt,
            temperature=0.0,
            max_tokens=128,
            model=self.enhancer_model,
        )

        text = response.content or ""
        return text.strip()
