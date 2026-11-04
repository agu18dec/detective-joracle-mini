"""One chat-completion call through OpenRouter, with a disk cache.

Any OpenRouter key works (CS329Z students have one). Identical requests are
answered from ``.cache/llm/``, so re-running an example or the eval costs
nothing. Set ``JORACLE_SALT`` to draw fresh samples instead.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "llm"
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"


def load_env() -> None:
    """Read KEY=VALUE lines from ./.env into the environment (no dependency)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


class LLM:
    def __init__(self, model: str | None = None) -> None:
        load_env()
        self.model = model or os.environ.get("JORACLE_MODEL", DEFAULT_MODEL)
        self.salt = os.environ.get("JORACLE_SALT", "")
        self.cost = 0.0
        self.calls = 0
        self._client = None

    def _connect(self) -> OpenAI:
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=5, timeout=180)
        return self._client

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4000) -> dict:
        """Returns the assistant message as a dict: {"content", "tool_calls"}."""
        request = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            request["tools"] = tools
        key = hashlib.sha256(json.dumps([request, self.salt], sort_keys=True).encode()).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        self.calls += 1
        if path.exists():
            return json.loads(path.read_text())

        for attempt in range(3):
            response = self._connect().chat.completions.create(**request, extra_body={"usage": {"include": True}})
            if response.choices:
                break
            time.sleep(2 * (attempt + 1))
        message = response.choices[0].message
        out = {
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in (message.tool_calls or [])
            ],
        }
        cost = getattr(response.usage, "cost", None)
        self.cost += float(cost) if isinstance(cost, (int, float)) else 0.0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out
