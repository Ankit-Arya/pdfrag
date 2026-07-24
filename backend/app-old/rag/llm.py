from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from app.config import get_settings
from app.rag.prompts import SYSTEM_PROMPT


class LlmConfigurationError(RuntimeError):
    pass


class LlmService:
    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            settings = get_settings()
            if not settings.openai_api_key and not settings.llm_base_url:
                raise LlmConfigurationError(
                    "No LLM is configured. Set OPENAI_API_KEY or an OpenAI-compatible LLM_BASE_URL."
                )
            kwargs: dict[str, str] = {
                "api_key": settings.openai_api_key or "local-not-required",
            }
            if settings.llm_base_url:
                kwargs["base_url"] = settings.llm_base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=4),
        reraise=True,
    )
    def answer(self, user_prompt: str) -> str:
        settings = get_settings()
        if settings.llm_base_url:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            return (content or "").strip()

        response = self.client.responses.create(
            model=settings.llm_model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
        return response.output_text.strip()


llm_service = LlmService()
