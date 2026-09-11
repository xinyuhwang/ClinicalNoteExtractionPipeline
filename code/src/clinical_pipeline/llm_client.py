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

DEFAULT_MODEL = "claude-sonnet-5"  # swap for claude-haiku-4-5, claude-opus-5, etc.

# Current Claude models (Sonnet 5, Opus 5, ...) removed the sampling parameters:
# passing `temperature`, `top_p`, or `top_k` is rejected with a 400. Reasoning
# depth is controlled through `output_config.effort` instead, which is what
# ensemble mode varies to get different runs out of the same note.
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class LLMResponse:
    result: ExtractionResult
    raw_json: dict
    latency_ms: float
    model: str


class LLMClient:
    """Structured-output client. One call = one validated ExtractionResult."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 16000,
                 effort: Optional[str] = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            ) from e
        import anthropic

        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"effort must be one of {VALID_EFFORTS}, got {effort!r}")

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        # Thinking is on by default on current models and shares the max_tokens
        # budget with the tool call, so this needs headroom -- too low a value
        # truncates the extraction rather than producing a short one.
        self.max_tokens = max_tokens
        self.effort = effort  # None = leave it to the API default

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
        effort: Optional[str] = None,
    ) -> LLMResponse:
        """Call the LLM once and return a validated ExtractionResult.

        `effort` overrides the client's default reasoning effort for this call
        (see VALID_EFFORTS). There is deliberately no `temperature` argument --
        it is rejected by the current models.
        """
        prompt = build_prompt(note_text, version=prompt_version, **(extra_prompt_kwargs or {}))

        effort = self.effort if effort is None else effort
        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"effort must be one of {VALID_EFFORTS}, got {effort!r}")

        tool_schema = self._extract_schema(ExtractionResult)
        tool = {
            "name": "record_extraction",
            "description": "Record the structured clinical findings extracted from the note.",
            "input_schema": tool_schema,
        }

        extra_params: dict = {}
        if effort is not None:
            extra_params["output_config"] = {"effort": effort}

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_extraction"},
            messages=[{"role": "user", "content": prompt}],
            **extra_params,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # A truncated response yields a partial tool call, which would surface
        # downstream as a confusing schema error. Fail with the real reason.
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Extraction truncated: hit max_tokens ({self.max_tokens}). "
                "Raise max_tokens or shorten the note."
            )

        tool_use_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        if tool_use_block is None:
            raise RuntimeError(f"No tool_use block in response: {response}")

        raw_json = tool_use_block.input
        result = ExtractionResult.model_validate(raw_json)

        return LLMResponse(result=result, raw_json=raw_json, latency_ms=latency_ms, model=self.model)

    def critique(self, critique_prompt: str) -> str:
        """Free-text critique call (no forced schema).

        Returns "" if the model produced no text block. Callers treat an empty
        critique as "nothing rejected", so this fails open -- a truncated or
        empty critique never drops a finding on its own.
        """
        response = self.client.messages.create(
            model=self.model,
            # Needs room for thinking tokens as well as the critique text.
            max_tokens=4000,
            messages=[{"role": "user", "content": critique_prompt}],
        )
        text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
        return text_block.text if text_block else ""
