"""LLM backends behind one small interface.

* ``OllamaClient`` - local, free (default). Run ``ollama pull qwen2.5:7b`` first.
* ``OpenAICompatibleClient`` - any /v1/chat/completions server (Groq free tier,
  LM Studio, vLLM, OpenAI). Lets you compare models in the evaluation.
* ``FakeLLM`` - deterministic stub for tests.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

import httpx


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
    def __init__(
        self, model: str, base_url: str, api_key: str | None = None, timeout: float = 120.0
    ):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.name = f"openai-compatible:{model}"

    def generate(self, system: str, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


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

    ``ollama:qwen2.5:7b`` | ``openai:<model>@<base_url>`` | ``fake``.
    Defaults to the ``ATHEX_LLM`` env var, then ``ollama:qwen2.5:7b``.
    """
    spec = spec or os.environ.get("ATHEX_LLM", "ollama:qwen2.5:7b")
    if spec == "fake":
        return FakeLLM()
    if spec.startswith("ollama:"):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaClient(spec.removeprefix("ollama:"), host=host)
    if spec.startswith("openai:"):
        model, _, base_url = spec.removeprefix("openai:").partition("@")
        return OpenAICompatibleClient(model, base_url or "https://api.openai.com/v1")
    raise ValueError(f"unknown LLM spec {spec!r}")
