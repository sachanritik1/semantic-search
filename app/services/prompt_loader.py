# app/services/prompt_loader.py

from pathlib import Path


PROMPT_DIR = Path(__file__).parent.parent / "prompt_templates"


def load_prompt(template_name: str) -> str:
    path = PROMPT_DIR / template_name
    if not path.exists():
        raise ValueError(f"Prompt template not found: {template_name}")
    return path.read_text()

def render_prompt(template: str, variables: dict[str, str]) -> str:
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return prompt
