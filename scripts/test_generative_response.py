import unittest

import torch

from log_analysis.models.generative import (
    THINKING_MIN_NEW_TOKENS,
    GenerativeConfig,
    GenerativeModel,
    _strip_reasoning,
)


class _OfflineGenerative(GenerativeModel):
    """GenerativeModel that never touches the network."""

    def instanceModelClient(self) -> None:  # noqa: D102
        pass


def _model(**overrides) -> _OfflineGenerative:
    config = GenerativeConfig(
        model_name="Qwen/Qwen3-0.6B",
        backend="huggingface",
        contextWindow=4096,  # avoids auto-detection (and any model download)
        **overrides,
    )
    return _OfflineGenerative(config)


class _FakeTokenizer:
    def __init__(self, reject_enable_thinking: bool = False, decoded: str = '{"is_error": true}') -> None:
        self.chat_template = "{{ messages }}"
        self.reject_enable_thinking = reject_enable_thinking
        self.decoded = decoded
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        if self.reject_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append(kwargs)
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.decoded


class _FakeHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.seen: dict | None = None

    def generate(self, **kwargs):
        self.seen = kwargs
        ids = kwargs["input_ids"]
        return torch.cat([ids, torch.zeros((ids.shape[0], 2), dtype=torch.long)], dim=1)


def _wire(model: _OfflineGenerative, tokenizer: _FakeTokenizer) -> _FakeHFModel:
    fake_model = _FakeHFModel()
    model._tokenizer = tokenizer
    model._model = fake_model
    return fake_model


class StripReasoningTest(unittest.TestCase):
    def test_removes_ascii_think_block(self):
        raw = '<think>\nreasoning here\n</think>\n{"is_error": true}'
        self.assertEqual(_strip_reasoning(raw), '{"is_error": true}')

    def test_removes_fullwidth_think_block(self):
        raw = "\uff1cthink\uff1e\nreasoning\n\uff1c/think\uff1e\n{\"is_error\": false}"
        self.assertEqual(_strip_reasoning(raw), '{"is_error": false}')

    def test_unclosed_think_block_keeps_the_json(self):
        raw = '<think>still reasoning {"is_error": true, "error_description": null}'
        self.assertEqual(_strip_reasoning(raw), '{"is_error": true, "error_description": null}')

    def test_removes_code_fences(self):
        raw = '```json\n{"is_error": true}\n```'
        self.assertEqual(_strip_reasoning(raw), '{"is_error": true}')

    def test_leaves_plain_json_untouched(self):
        raw = '  {"is_error": true}  '
        self.assertEqual(_strip_reasoning(raw), '{"is_error": true}')


class ParseResponseTest(unittest.TestCase):
    def test_parses_fenced_json(self):
        data = _model()._parse_response('```json\n{"is_error": true, "error_description": "db down"}\n```')
        self.assertTrue(data["is_valid_response"])
        self.assertTrue(data["is_error"])

    def test_prefers_the_answer_over_an_example_object(self):
        raw = 'Example: {"foo": 1} ... real answer: {"is_error": false, "error_description": null}'
        data = _model()._parse_response(raw)
        self.assertTrue(data["is_valid_response"])
        self.assertFalse(data["is_error"])
        self.assertNotIn("foo", data)

    def test_reasoning_without_json_is_invalid(self):
        # This is the Qwen3 failure mode when thinking is left on and the budget runs out.
        raw = "<think>\nOkay, let's see. The user provided two logs: [0] ERROR: database connection timed out"
        data = _model()._parse_response(raw)
        self.assertFalse(data["is_valid_response"])
        self.assertNotIn("is_error", data)


class CallHuggingFaceTest(unittest.TestCase):
    def test_thinking_off_is_passed_to_the_chat_template(self):
        model = _model(thinking=False, max_new_tokens=64)
        tokenizer = _FakeTokenizer()
        fake_model = _wire(model, tokenizer)

        model._call_hf()

        self.assertIs(tokenizer.calls[0]["enable_thinking"], False)
        self.assertIs(tokenizer.calls[0]["add_generation_prompt"], True)
        self.assertEqual(fake_model.seen["max_new_tokens"], 64)

    def test_thinking_on_uses_the_configured_flag_and_a_larger_budget(self):
        model = _model(thinking=True, max_new_tokens=64)
        tokenizer = _FakeTokenizer()
        fake_model = _wire(model, tokenizer)

        model._call_hf()

        self.assertIs(tokenizer.calls[0]["enable_thinking"], True)
        self.assertEqual(fake_model.seen["max_new_tokens"], THINKING_MIN_NEW_TOKENS)

    def test_falls_back_when_template_rejects_enable_thinking(self):
        model = _model(thinking=False)
        tokenizer = _FakeTokenizer(reject_enable_thinking=True)
        _wire(model, tokenizer)

        text, _ = model._call_hf()

        self.assertNotIn("enable_thinking", tokenizer.calls[0])
        self.assertEqual(tokenizer.calls[0]["add_generation_prompt"], True)
        self.assertEqual(text, '{"is_error": true}')


if __name__ == "__main__":
    unittest.main()
