from typing import Optional


class UniversalAIClient:
    """
    Lightweight wrapper around OpenAI and Anthropic clients.

    This keeps the same provider logic as the original notebooks without adding
    retry logic, moderation, or other behavioral changes.
    """

    def __init__(self):
        self.provider: Optional[str] = None
        self.client = None
        self.model_name: Optional[str] = None

    def setup_openai(self, api_key: str, model: str = "gpt-5.1") -> None:
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.provider = "openai"
        self.model_name = model

    def setup_anthropic(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.provider = "anthropic"
        self.model_name = model

    def call_ai(self, prompt: str, system_prompt: str) -> str:
        if self.client is None or self.provider is None or self.model_name is None:
            raise RuntimeError("AI client not configured. Call setup_openai or setup_anthropic first.")

        if self.provider == "openai":
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return res.choices[0].message.content

        if self.provider == "anthropic":
            res = self.client.messages.create(
                model=self.model_name,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return res.content[0].text

        raise RuntimeError(f"Unsupported provider: {self.provider}")


def build_variation_prompt(prompt: str, index: int) -> str:
    """
    Format prompt variants exactly as done in the Zero-Box pilot.
    """

    return f"{prompt}\n\n(Answer variation {index})"
