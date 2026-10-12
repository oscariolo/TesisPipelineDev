import math
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

        text, _, _ = model._call_hf()

        self.assertNotIn("enable_thinking", tokenizer.calls[0])
        self.assertEqual(tokenizer.calls[0]["add_generation_prompt"], True)
        self.assertEqual(text, '{"is_error": true}')


class _StubTokenizer:
    """Minimal tokenizer exposing only the pieces the decision scan needs."""

    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self._pieces = {'"': 1, "is_error": 2, ": ": 3, " true": 10, " false": 11}
        self._ids = {value: key for key, value in self._pieces.items()}
        self.vocab_size = 64

    def encode(self, text, add_special_tokens: bool = False):
        return [self._pieces[text]] if text in self._pieces else []

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._ids.get(token_id, "?") for token_id in ids)

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


def _logits(true_logit: float = 0.0, false_logit: float = 0.0) -> list[float]:
    row = [0.0] * 64
    row[10] = true_logit   # " true"
    row[11] = false_logit  # " false"
    return row


class DecisionProbabilityTest(unittest.TestCase):
    def _model_with(self, tokenizer: _StubTokenizer) -> _OfflineGenerative:
        model = _model()
        model._tokenizer = tokenizer
        return model

    def test_probability_comes_from_the_boolean_after_the_is_error_key(self):
        model = self._model_with(_StubTokenizer())
        ids = [1, 2, 3, 10]                                     # "is_error": true
        scores = [_logits(), _logits(), _logits(), _logits(2.0, 0.0)]

        probability = model._decision_probability(scores, ids)

        self.assertAlmostEqual(probability, 1.0 / (1.0 + math.exp(-2.0)))

    def test_boolean_before_the_key_is_only_a_fallback(self):
        model = self._model_with(_StubTokenizer())
        ids = [11, 1, 2, 3, 10]                                 # a stray false, then "is_error": true
        scores = [_logits(10.0, -10.0), _logits(), _logits(), _logits(), _logits(0.0, 2.0)]

        probability = model._decision_probability(scores, ids)

        # The stray token would give ~1.0; the real answer after the key gives ~0.119.
        self.assertAlmostEqual(probability, 1.0 / (1.0 + math.exp(2.0)))

    def test_falls_back_to_the_first_boolean_when_the_key_is_absent(self):
        model = self._model_with(_StubTokenizer())
        ids = [1, 10]                                          # bare true, no is_error key
        scores = [_logits(), _logits(1.0, -1.0)]

        probability = model._decision_probability(scores, ids)

        self.assertAlmostEqual(probability, 1.0 / (1.0 + math.exp(-2.0)))

    def test_no_boolean_token_means_no_probability(self):
        model = self._model_with(_StubTokenizer())
        self.assertIsNone(model._decision_probability([_logits()], [1]))

    def test_no_scores_means_no_probability(self):
        model = self._model_with(_StubTokenizer())
        self.assertIsNone(model._decision_probability(None, [1, 2, 3, 10]))


class _LogitsModel(torch.nn.Module):
    """Stand-in that answers a forward pass with one canned logits row."""

    def __init__(self, row: list[float] | None = None) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.row = row
        self.calls: list[torch.Tensor] = []

    def forward(self, ids):
        self.calls.append(ids)
        if self.row is None:
            raise RuntimeError("forward pass must not be needed here")
        return type("_Out", (), {"logits": torch.tensor([[self.row]])})()


class MaskedDecisionTest(unittest.TestCase):
    def _model_with(self, tokenizer: _StubTokenizer) -> _OfflineGenerative:
        model = _model()
        model._tokenizer = tokenizer
        return model

    def test_masked_groups_are_re_read_with_one_raw_forward_pass(self):
        model = self._model_with(_StubTokenizer())
        # The logits warpers wiped both token groups, so the recorded row is useless.
        masked = [_logits(), _logits(), _logits(), _logits(-math.inf, -math.inf)]
        raw = _LogitsModel(_logits(3.0, -3.0))
        model._model = raw

        probability = model._decision_probability(
            masked, [1, 2, 3, 10], sequences=torch.zeros((1, 8), dtype=torch.long), prompt_length=4
        )

        self.assertEqual(len(raw.calls), 1)
        self.assertAlmostEqual(probability, 1.0 / (1.0 + math.exp(-6.0)))

    def test_a_usable_score_skips_the_extra_forward_pass(self):
        model = self._model_with(_StubTokenizer())
        raw = _LogitsModel(None)  # raises if called
        model._model = raw

        probability = model._decision_probability(
            [_logits(), _logits(), _logits(), _logits(2.0, 0.0)],
            [1, 2, 3, 10],
            sequences=torch.zeros((1, 8), dtype=torch.long),
            prompt_length=4,
        )

        self.assertEqual(raw.calls, [])
        self.assertAlmostEqual(probability, 1.0 / (1.0 + math.exp(-2.0)))

    def test_masked_groups_without_a_prefix_give_no_probability(self):
        model = self._model_with(_StubTokenizer())
        masked = [_logits(), _logits(), _logits(), _logits(-math.inf, -math.inf)]

        probability = model._decision_probability(masked, [1, 2, 3, 10])

        self.assertIsNone(probability)


if __name__ == "__main__":
    unittest.main()
