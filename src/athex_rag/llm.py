"""LLM backends behind one small interface.

* ``OllamaClient`` - local, free (default). Run ``ollama pull qwen2.5:7b`` first.
* ``OpenAICompatibleClient`` - any /v1/chat/completions server (``groq:<model>``,
  LM Studio, vLLM, OpenAI). Lets you compare models in the evaluation.
* ``FakeLLM`` - deterministic stub for tests.
"""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
    ):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"

    def generate(self, system: str, prompt: str) -> str:
        response = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class OpenAICompatibleClient:
    """Any /v1/chat/completions API (Groq, OpenAI, LM Studio, vLLM...).

    Free tiers (e.g. Groq) rate-limit by tokens per minute; a 429 or 5xx response is
    retried after the server's ``Retry-After`` delay (or exponential backoff).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 8,
        name_prefix: str = "openai-compatible",
    ):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.max_retries = max_retries
        self.name = f"{name_prefix}:{model}"

    def _post(self, payload: dict) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )

    def generate(self, system: str, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not set (see README: Using a hosted LLM)")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        for attempt in range(self.max_retries + 1):
            response = self._post(payload)
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_retries:
                break
            try:
                delay = float(response.headers.get("retry-after", ""))
            except ValueError:
                delay = 2.0 * 2**attempt
            time.sleep(min(delay, 60.0))
        if response.status_code >= 400:
            raise RuntimeError(
                f"{self.name}: HTTP {response.status_code} from {self.base_url}: "
                f"{response.text[:400]}"
            )
        return strip_reasoning(response.json()["choices"][0]["message"]["content"] or "")


def strip_reasoning(text: str) -> str:
    """Drop a reasoning block (``<think>...</think>``) that some models (Qwen, DeepSeek)
    put before the answer. Its numbers and [n] markers would otherwise be scored as if
    they were part of the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # opening tag missing / truncated
        text = text.split("</think>")[-1]
    return text.strip()


def check_llm(llm: LLMClient) -> None:
    """Make one tiny request so a bad model name or key fails in seconds, not after
    the retrieval stages have run."""
    llm.generate("Απάντησε μόνο με τη λέξη OK.", "OK;")


class FakeLLM:
    """Echoes the first source back with a citation; says 'not found' if none."""

    name = "fake"

    def generate(self, system: str, prompt: str) -> str:
        match = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n\[2\]|\n\nΕρώτηση:)", prompt, re.S)
        if not match:
            return "Δεν βρέθηκε η πληροφορία στα έγγραφα."
        return f"{match.group(1).strip()[:300]} [1]"


def make_llm(spec: str | None = None) -> LLMClient:
    """Build a client from a spec string.

    ``ollama:qwen2.5:7b`` | ``groq:<model>`` | ``openai:<model>@<base_url>`` | ``fake``.
    Hosted APIs read the key from the ``LLM_API_KEY`` environment variable.
    Defaults to the ``ATHEX_LLM`` env var, then ``ollama:qwen2.5:7b``.
    """
    spec = spec or os.environ.get("ATHEX_LLM", "ollama:qwen2.5:7b")
    if spec == "fake":
        return FakeLLM()
    if spec.startswith("ollama:"):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaClient(spec.removeprefix("ollama:"), host=host)
    # a web demo should fail fast instead of waiting out a long rate limit
    retries = int(os.environ.get("ATHEX_LLM_MAX_RETRIES", "8"))
    if spec.startswith("groq:"):
        return OpenAICompatibleClient(
            spec.removeprefix("groq:"), GROQ_BASE_URL, name_prefix="groq", max_retries=retries
        )
    if spec.startswith("openai:"):
        model, _, base_url = spec.removeprefix("openai:").partition("@")
        return OpenAICompatibleClient(
            model, base_url or "https://api.openai.com/v1", max_retries=retries
        )
    raise ValueError(f"unknown LLM spec {spec!r}")
