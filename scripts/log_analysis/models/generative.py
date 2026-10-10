import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult
from log_analysis.models.base import BaseModel
import ollama

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
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        logger.warning("PyTorch not installed, cannot determine VRAM.")
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

    def _call_hf(self, batch_size: int = 0) -> tuple[str, int]:
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
        outputs = self._model.generate(**inputs, **gen_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated_length = outputs[0].shape[-1] - prompt_length
        text = self._tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        return text, prompt_length + generated_length

    def _call_ollama(self) -> tuple[str, int]:
        response = self._model.chat(
            model=self.config.model_name,
            messages=self._messages,
            think=self.config.thinking,
            options={
                "num_ctx": self._context_window,
            }
        )
        token_usage = response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        return response["message"]["content"].strip(), token_usage

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
        batch_text = "\n".join(f"[{entry.index}] {entry.raw_text}" for entry in batch.entries)

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
            raw_output, self._token_usage = self._call_ollama()
        else:
            raw_output, self._token_usage = self._call_hf()


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
            model_name=self.config.model_name if hasattr(self.config, "model_name") else None,
            token_usage=self._token_usage,
            is_valid_response=is_valid_response,
            raw_response=raw_output,
            error_description=data.get("error_description"),
            recommended_action=data.get("recommended_action"),
        )
