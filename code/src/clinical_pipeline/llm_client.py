"""
Thin wrapper around the Anthropic API that forces structured output via
tool-use, matching a Pydantic schema. Swap `model` as needed.

Requires: pip install anthropic
Requires: ANTHROPIC_API_KEY in your environment (see .env.example).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel

from .models import ExtractionResult
from .prompts import SYSTEM_PROMPT, build_prompt

DEFAULT_MODEL = "claude-sonnet-5"  # swap for claude-haiku-4-5-20251001, claude-opus-5, etc.


@dataclass
class LLMResponse:
    result: ExtractionResult
    raw_json: dict
    latency_ms: float
    model: str


class LLMClient:
    """Structured-output client. One call = one validated ExtractionResult."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1500, temperature: float = 0.0):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            ) from e
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _extract_schema(self, schema: Type[BaseModel]) -> dict:
        """Turn a Pydantic model into an Anthropic tool input_schema."""
        js = schema.model_json_schema()
        js.pop("title", None)
        return js

    def extract(
        self,
        note_text: str,
        prompt_version: str = "v1",
        extra_prompt_kwargs: Optional[dict] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Call the LLM once and return a validated ExtractionResult."""
        prompt = build_prompt(note_text, version=prompt_version, **(extra_prompt_kwargs or {}))

        tool_schema = self._extract_schema(ExtractionResult)
        tool = {
            "name": "record_extraction",
            "description": "Record the structured clinical findings extracted from the note.",
            "input_schema": tool_schema,
        }

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_extraction"},
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        tool_use_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        if tool_use_block is None:
            raise RuntimeError(f"No tool_use block in response: {response}")

        raw_json = tool_use_block.input
        result = ExtractionResult.model_validate(raw_json)

        return LLMResponse(result=result, raw_json=raw_json, latency_ms=latency_ms, model=self.model)

    def critique(self, critique_prompt: str) -> str:
        """Free-text critique call (no forced schema)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": critique_prompt}],
        )
        text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
        return text_block.text if text_block else ""
