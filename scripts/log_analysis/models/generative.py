import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult, render_batch_text
from log_analysis.models.base import BaseModel
import ollama
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a log analysis assistant. You will be given a group of log entries. "
        "Determine if there is an error in this group. "
        "Reply with raw JSON only, in exactly this shape:\n"
        '{"is_error": true or false, "error_description": "brief description or null", '
        '"recommended_action": "what to do or null"}\n'
        "Do not add explanations, markdown code fences, or any text before or after the JSON."
    ),
}

# Reasoning blocks emitted by thinking models (Qwen3 wraps them in fullwidth angle brackets).
THINK_BLOCK_RE = re.compile(
    r"[<\uff1c]\s*think(?:ing)?\s*[>\uff1e].*?[<\uff1c]\s*/\s*think(?:ing)?\s*[>\uff1e]",
    re.DOTALL | re.IGNORECASE,
)
THINK_OPEN_RE = re.compile(r"[<\uff1c]\s*think(?:ing)?\s*[>\uff1e]", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*")

# Thinking mode needs a much larger completion budget than a single JSON object.
THINKING_MIN_NEW_TOKENS = 1024

# Forms the JSON boolean can take; only single-token spellings are usable as decision logits.
TRUE_FORMS = (" true", "true", " True", "TRUE")
FALSE_FORMS = (" false", "false", " False", "FALSE")


def _resolve_boolean_token_ids(tokenizer) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Single-token ids that can spell ``true`` / ``false`` in this tokenizer's vocabulary.

    A tokenizer that cannot be interrogated yields empty tuples, which makes the caller report an
    unknown score instead of failing the batch.
    """
    def single_token_ids(forms: tuple[str, ...]) -> tuple[int, ...]:
        encoder = getattr(tokenizer, "encode", None)
        if encoder is None:
            return ()
        ids = []
        for form in forms:
            encoded = encoder(form, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
        return tuple(sorted(set(ids)))

    return single_token_ids(TRUE_FORMS), single_token_ids(FALSE_FORMS)


def _logits_tensor(row) -> torch.Tensor:
    """A decoding step's logits as a float32 tensor (an existing float32 tensor is passed through)."""
    if isinstance(row, torch.Tensor):
        return row if row.dtype is torch.float32 else row.to(torch.float32)
    return torch.as_tensor(row, dtype=torch.float32)


def _step_row(step) -> torch.Tensor:
    """One decoding step's logits as a flat row (generate emits ``[batch, vocab]``)."""
    row = _logits_tensor(step)
    return row[0] if row.dim() > 1 else row


def _group_is_usable(row, ids: tuple[int, ...]) -> bool:
    """Whether a token group survived the logits warpers (``generate`` masks the rest with -inf)."""
    return bool(torch.isfinite(_logits_tensor(row)[list(ids)]).any())


def _two_way_probability(logits_row, true_ids: tuple[int, ...], false_ids: tuple[int, ...]) -> float:
    """P(true) / (P(true) + P(false)) at one decoding step, from that step's raw logits.

    The group logits are reduced with ``torch.logsumexp`` and normalised with ``torch.softmax`` so
    multi-token spellings of the same boolean sum their probability mass.
    """
    logits = _logits_tensor(logits_row)
    group_logits = torch.stack(
        [
            torch.logsumexp(logits[list(true_ids)], dim=-1),
            torch.logsumexp(logits[list(false_ids)], dim=-1),
        ]
    )
    return float(torch.softmax(group_logits, dim=-1)[0])


def _strip_reasoning(text: str) -> str:
    """Drop reasoning blocks and markdown fences so only the JSON answer is left."""
    text = CODE_FENCE_RE.sub("", text)
    text = THINK_BLOCK_RE.sub(" ", text)
    unclosed = THINK_OPEN_RE.search(text)
    if unclosed:
        brace = text.find("{", unclosed.end())
        text = text[brace:] if brace != -1 else text[: unclosed.start()]
    return text.strip()

def getMaxVramAvailable() -> int:
    """Returns the maximum VRAM available on the current device in bytes."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
    except Exception:
        logger.warning("Could not query the device, cannot determine VRAM.")
    return 0


class GenerativeConfig(PydanticBaseModel):
    model_name: str
    backend: str
    keepHistory: Optional[str] = "perPrompt" #modes: perPrompt, tokenLimit, always (default behaviour is perPrompt)
    tokenLimitPercentage: Optional[float] = 1.0 #only applies when keepHistory is tokenLimit, if the token usage exceeds this percentage of the context window, the history will be cleared for the next batch
    contextWindow: Optional[int] = None #sets the context window of the model, by default it will try to get the maximum available from the model
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    hf_device: str = "cpu"
    thinking: bool = False
    max_new_tokens: int = 512  # completion budget per batch; thinking mode needs much more
    bit_precision: Optional[str] = None  # e.g., "fp16", "int8", etc. for Hugging Face models
    tokenizer_name: Optional[str] = None  # e.g., base model repo when loading a GGUF model
    gguf_file: Optional[str] = None  # e.g., "Qwen3.6-27B-Q4_K_M.gguf" for GGUF repos



class GenerativeModel(BaseModel):
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._messages = []  # For chat history if applies
        self._token_usage = 0
        self._true_ids: tuple[int, ...] | None = None
        self._false_ids: tuple[int, ...] | None = None
        self._context_window = config.contextWindow
        self.instanceModelClient()
        # Resolve the window after the client exists so the HF backend can read the model config.
        if self._context_window is None:
            self._context_window = self._getMaxContextWindow()


    def instanceModelClient(self):
        if self.config.backend == "ollama": #instance on ollama client
            host = f"http://{self.config.ollama_host}:{self.config.ollama_port}"
            self._model = ollama.Client(host=host)
        if self.config.backend == "huggingface": #instance on huggingface client
            self._load_hf()

    def _getMaxContextWindow(self) -> int:
        if self.config.backend == "ollama":
            host = f"http://{self.config.ollama_host}:{self.config.ollama_port}"
            client = ollama.Client(host=host)
            details = client.show(model=self.config.model_name)
            maxWindow = details.get("modelinfo", {}).get("llama.context_length", 4096)
            return maxWindow
        if self.config.backend == "huggingface":
            # Prefer the model's own limit; tokenizer.model_max_length can be a huge sentinel.
            model_config = getattr(self._model, "config", None)
            max_positions = getattr(model_config, "max_position_embeddings", None)
            if max_positions:
                return int(max_positions)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name, gguf_file=self.config.gguf_file
                )
            return int(self._tokenizer.model_max_length)
        return 4096

    def _load_hf(self):
        logger.info("Loading HF model %s on %s", self.config.model_name, self.config.hf_device)
        #gguf file should take priority when determining tokenizer
        if self.config.gguf_file is not None:
            tokenizer_name = self.config.model_name
        else:
            tokenizer_name = self.config.tokenizer_name or self.config.model_name

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, gguf_file=self.config.gguf_file
            )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            gguf_file=self.config.gguf_file,
            offload_folder="./offloading",
        )

    def _boolean_ids(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Token ids usable as the true/false decision, resolved from the tokenizer once."""
        if self._true_ids is None or self._false_ids is None:
            self._true_ids, self._false_ids = _resolve_boolean_token_ids(self._tokenizer)
        return self._true_ids, self._false_ids

    def _decision_index(self, generated_ids: list[int]) -> int | None:
        """Position of the token that spells the answer boolean, or ``None`` if there is none.

        Walks the generated tokens keeping track of whether the ``is_error`` key has been seen, and
        takes the first boolean token after it — falling back to the first boolean token at all, which
        is what a model that answers without the key gives us.
        """
        true_ids, false_ids = self._boolean_ids()
        if not true_ids or not false_ids:
            return None

        decoded_so_far = ""
        seen_key = False
        fallback = None
        for index, token_id in enumerate(generated_ids):
            decoded_so_far += self._tokenizer.decode([token_id])
            if token_id in true_ids or token_id in false_ids:
                if seen_key:
                    return index
                if fallback is None:
                    fallback = index
            if not seen_key and "is_error" in decoded_so_far:
                seen_key = True

        return fallback

    def _raw_logits_at(self, sequences, position: int):
        """Raw (unmasked) logits for ``position``, via one extra forward pass.

        ``generate(output_scores=True)`` records the logits *after* the logits warpers, which the
        model's generation config fills with -inf (Qwen3 ships ``temperature=0.6, top_k=20,
        top_p=0.95``), destroying half the comparison. Re-running the prefix once recovers the real
        distribution without touching how the text itself was generated.
        """
        try:
            with torch.no_grad():
                outputs = self._model(sequences[:, :position])
        except Exception:  # a scoring failure must never fail the batch
            logger.warning("Could not read raw decision logits; leaving error_probability empty")
            return None
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits[0, -1]

    def _decision_probability(self, scores, generated_ids: list[int], sequences=None,
                              prompt_length: int = 0) -> float | None:
        """P(error=True) at the token that spells the JSON boolean.

        Prefers the recorded per-step scores; when the warpers masked a whole token group there, the
        prefix is re-forwarded once for the raw logits. Returns ``None`` when the batch produced no
        boolean token or no usable logits.
        """
        true_ids, false_ids = self._boolean_ids()
        if not true_ids or not false_ids:
            return None

        index = self._decision_index(generated_ids)
        if index is None:
            return None

        if scores is not None and index < len(scores):
            row = _step_row(scores[index])
            if _group_is_usable(row, true_ids) and _group_is_usable(row, false_ids):
                return _two_way_probability(row, true_ids, false_ids)

        if sequences is None or prompt_length == 0:
            return None
        row = self._raw_logits_at(sequences, prompt_length + index)
        if row is None or not _group_is_usable(row, true_ids) or not _group_is_usable(row, false_ids):
            return None
        return _two_way_probability(row, true_ids, false_ids)

    def _call_hf(self, batch_size: int = 0) -> tuple[str, int, float | None]:
        if getattr(self._tokenizer, "chat_template", None):
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
     
            try:
                inputs = self._tokenizer.apply_chat_template(
                    self._messages, enable_thinking=self.config.thinking, **template_kwargs
                )
            except TypeError:
                inputs = self._tokenizer.apply_chat_template(self._messages, **template_kwargs)
        else:
            prompt = "\n\n".join(
                f"{message['role'].upper()}:\n{message['content']}"
                for message in self._messages
            )
            prompt += "\n\nASSISTANT:\n"
            inputs = self._tokenizer(prompt, return_tensors="pt")
        if not isinstance(inputs, Mapping):
            raise TypeError("Hugging Face tokenizer must return a mapping")
        inputs = dict(inputs)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Safety: cap max_new_tokens to avoid unbounded generation OOM
        max_new_tokens = self.config.max_new_tokens
        if self.config.thinking and max_new_tokens < THINKING_MIN_NEW_TOKENS:
            logger.warning(
                "max_new_tokens=%d is too small for thinking mode; using %d so the answer is not truncated",
                max_new_tokens,
                THINKING_MIN_NEW_TOKENS,
            )
            max_new_tokens = THINKING_MIN_NEW_TOKENS
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        # output_scores gives the per-step logits we need for the true/false decision probability.
        outputs = self._model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs,
        )
        sequences = getattr(outputs, "sequences", outputs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = sequences[0][prompt_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        error_probability = self._decision_probability(
            getattr(outputs, "scores", None),
            generated.tolist(),
            sequences=sequences,
            prompt_length=prompt_length,
        )
        return text, prompt_length + len(generated), error_probability

    def _call_ollama(self) -> tuple[str, int, None]:
        response = self._model.chat(
            model=self.config.model_name,
            messages=self._messages,
            think=self.config.thinking,
            options={
                "num_ctx": self._context_window,
            }
        )
        token_usage = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        # Ollama exposes no comparable logits, so this backend leaves the score unknown.
        return response["message"]["content"].strip(), token_usage, None

    def _parse_response(self, text: str) -> dict:
        text = _strip_reasoning(text)
        decoder = json.JSONDecoder()
        fallback: dict | None = None
        for match in re.finditer(r"\{", text):
            candidate = text[match.start():]
            try:
                json_result, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                # Some models emit a closing parenthesis instead of the JSON brace.
                if not candidate.rstrip().endswith(")"):
                    continue
                try:
                    json_result = json.loads(candidate.rstrip()[:-1] + "}")
                except json.JSONDecodeError:
                    continue
            if isinstance(json_result, dict):
                json_result["is_valid_response"] = True
                # Prefer the real answer over an example object quoted inside prose.
                if "is_error" in json_result:
                    return json_result
                if fallback is None:
                    fallback = json_result

        if fallback is not None:
            return fallback
        logger.warning("Failed to parse JSON from model output: %.200s", text)
        return {"is_valid_response": False}



    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        batch_text = render_batch_text(batch)

        #cuando es perPrompt, se limpia historial de mensajes
        if self.config.keepHistory == "perPrompt":
            self._messages = [SYSTEM_PROMPT]
            self._messages.append({
                "role":"user",
                "content": batch_text
            })

        #cuando es por tokenLimit, se limpia historial de mensajes (los primeros mensajes) si se supera el limite de tokens
        if self.config.keepHistory == "tokenLimit" or self.config.keepHistory == "always":
            if not self._messages:
                self._messages.append(SYSTEM_PROMPT)
            self._messages.append({
                "role": "user",
                "content": batch_text,
            })

        start = time.perf_counter()

        if self.config.backend == "ollama":
            raw_output, self._token_usage, error_probability = self._call_ollama()
        else:
            raw_output, self._token_usage, error_probability = self._call_hf()


        elapsed = time.perf_counter() - start
        logger.info("LLM batch #%d analyzed in %.3fs", batch.batch_id, elapsed)

        data = self._parse_response(raw_output)
        
        is_valid_response = bool(data.get("is_valid_response", False))

        if is_valid_response:
            error_found = bool(data.get("is_error", False))

        logger.info("Current token usage: %d, context window: %d", self._token_usage, self._context_window)

        #tokenLimit mode: si supera un porcentaje de tokens se limpia historial para el siguiente batch (reinicio de contexto)
        if (
            self.config.keepHistory == "tokenLimit"
            and self._context_window
            and self.config.tokenLimitPercentage
            and self._token_usage >= self._context_window * self.config.tokenLimitPercentage
        ):
            self._messages = []
            self._token_usage = 0

    
        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=error_found,
            error_probability=error_probability,
            model_name=self.config.model_name if hasattr(self.config, "model_name") else None,
            token_usage=self._token_usage,
            is_valid_response=is_valid_response,
            raw_response=raw_output,
            error_description=data.get("error_description"),
            recommended_action=data.get("recommended_action"),
        )
