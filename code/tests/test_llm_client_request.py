"""
Tests the request LLMClient actually sends, with a stubbed `anthropic` module.

No API key and no installed SDK required. This exists because the real client
is the one component MockLLMClient cannot cover, and its request shape is
version-sensitive: sampling parameters (`temperature`, `top_p`, `top_k`) were
removed from the current Claude models and are rejected with a 400, so a
well-meaning `temperature=0.0` for "determinism" breaks every extraction call.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CAPTURED: dict = {}


class _FakeToolUseBlock:
    type = "tool_use"
    input = {
        "findings": [{"condition": "Diabetes", "evidence": "diabetes", "confidence": 0.9}],
        "note_summary": "summary",
    }


class _FakeTextBlock:
    type = "text"
    text = "Chest Pain is unsupported."


class _FakeResponse:
    content = [_FakeToolUseBlock()]
    stop_reason = "tool_use"


class _FakeMessages:
    def create(self, **kwargs):
        CAPTURED.clear()
        CAPTURED.update(kwargs)
        return _FakeResponse()


class _FakeAnthropic:
    def __init__(self, *args, **kwargs):
        self.messages = _FakeMessages()


_fake_module = types.ModuleType("anthropic")
_fake_module.Anthropic = _FakeAnthropic
sys.modules.setdefault("anthropic", _fake_module)

from clinical_pipeline.llm_client import VALID_EFFORTS, LLMClient  # noqa: E402

SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def test_request_sends_no_sampling_parameters():
    LLMClient().extract("Patient has diabetes.")
    for param in SAMPLING_PARAMS:
        assert param not in CAPTURED, f"{param} is rejected by current models"


def test_extraction_still_forces_the_tool_call():
    LLMClient().extract("Patient has diabetes.")
    assert CAPTURED["tool_choice"] == {"type": "tool", "name": "record_extraction"}
    assert CAPTURED["tools"][0]["name"] == "record_extraction"


def test_max_tokens_leaves_headroom_for_thinking_tokens():
    # Thinking is on by default on current models and shares this budget with
    # the tool call, so a small value truncates extractions.
    LLMClient().extract("Patient has diabetes.")
    assert CAPTURED["max_tokens"] >= 16000


def test_effort_is_sent_as_output_config_and_is_optional():
    LLMClient().extract("Patient has diabetes.")
    assert "output_config" not in CAPTURED  # unset -> leave the API default alone

    LLMClient().extract("Patient has diabetes.", effort="low")
    assert CAPTURED["output_config"] == {"effort": "low"}

    LLMClient(effort="xhigh").extract("Patient has diabetes.")
    assert CAPTURED["output_config"] == {"effort": "xhigh"}


def test_invalid_effort_is_rejected_before_the_api_call():
    for bad in ("bogus", "hot", "0.4"):
        try:
            LLMClient(effort=bad)
        except ValueError:
            continue
        raise AssertionError(f"effort={bad!r} should be rejected; valid: {VALID_EFFORTS}")


def test_truncated_response_raises_instead_of_failing_schema_validation():
    client = LLMClient()
    _FakeResponse.stop_reason = "max_tokens"
    try:
        client.extract("Patient has diabetes.")
    except RuntimeError as e:
        assert "max_tokens" in str(e)
    else:
        raise AssertionError("a truncated tool call should raise a clear error")
    finally:
        _FakeResponse.stop_reason = "tool_use"


def test_critique_call_sends_no_sampling_parameters():
    client = LLMClient()
    _FakeResponse.content = [_FakeTextBlock()]
    try:
        assert client.critique("review this") == "Chest Pain is unsupported."
        for param in SAMPLING_PARAMS:
            assert param not in CAPTURED
    finally:
        _FakeResponse.content = [_FakeToolUseBlock()]


if __name__ == "__main__":
    test_request_sends_no_sampling_parameters()
    test_extraction_still_forces_the_tool_call()
    test_max_tokens_leaves_headroom_for_thinking_tokens()
    test_effort_is_sent_as_output_config_and_is_optional()
    test_invalid_effort_is_rejected_before_the_api_call()
    test_truncated_response_raises_instead_of_failing_schema_validation()
    test_critique_call_sends_no_sampling_parameters()
    print("All llm_client request tests passed.")
