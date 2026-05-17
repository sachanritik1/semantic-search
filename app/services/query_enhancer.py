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
            "Rewrite this search query to improve document retrieval.\n"
            "Rules:\n"
            "- Keep the same intent and entities as the original.\n"
            "- Do not invent facts or assumptions (e.g. do not add 'public figure' "
            "unless the user said so).\n"
            "- Add only neutral clarifiers or synonyms useful for search.\n\n"
            f"Original query: \"{query}\"\n\n"
            "Return the enhanced query only, no other text."
        )
        

        response = self.llm_service.generate_text(
            prompt,
            temperature=0.0,
            max_tokens=128,
            model=self.enhancer_model,
        )

        text = response.content or ""
        return text.strip()
