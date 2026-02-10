from collections import Counter

from app.services.llm_service import LLMService

def extract_final_answer(content: str) -> str:
    """
    Placeholder function to extract the final answer from the LLM response content which contains the chain of thought reasoning with final answer after "FINAL_ANSWER:"
    """
    marker = "FINAL_ANSWER:"
    
    if marker not in content:
        return content.strip()  # Fallback to returning the whole content if marker is not found

    answer = content.split(marker, 1)[1]
    return answer.strip()


async def generate_with_self_consistency(
    llm_service: LLMService,
    prompt: str,
    runs: int = 3,
):
    answers = []

    for _ in range(runs):
        response = await llm_service.generate_text_async(
            prompt,
            temperature=0.9,
        )
        final = extract_final_answer(response.content)
        answers.append(final)

        print(f"Run {_+1}: Final answer extracted: {final}")

    return Counter(answers).most_common(1)[0][0]
