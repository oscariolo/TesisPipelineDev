import json
import logging
import re
import time
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult
from log_analysis.models.base import BaseModel

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Determine if there is an error in this group of logs.Respond ONLY with valid JSON in this exact format, do not include any extra text or explanations:
{{
  "is_error": true or false,
  "error_description": "brief description of the error or null",
  "recommended_action": "what action to take or null"
}}

Log entry: {raw_text}"""


class GenerativeConfig(PydanticBaseModel):
    model_name: str
    backend: str
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    hf_device: str = "cpu"
    thinking: bool = False
    token: str = None  # Optional API token for Ollama or other services
    bit_precision: Optional[str] = None  # e.g., "fp16", "int8", etc. for Hugging Face models
    tokenizer_name: Optional[str] = None  # e.g., base model repo when loading a GGUF model
    gguf_file: Optional[str] = None  # e.g., "Qwen3.6-27B-Q4_K_M.gguf" for GGUF repos



class GenerativeModel(BaseModel):
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_hf(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HF model %s on %s", self.config.model_name, self.config.hf_device)
        kwargs = {"token": self.config.token}
        if self.config.gguf_file:
            kwargs["gguf_file"] = self.config.gguf_file
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.hf_device,
            **kwargs
        )

    def _call_hf(self, prompt: str) -> str:
        if self._model is None:
            self._load_hf()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model.generate(**inputs, max_new_tokens=200)
        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()

    def _call_ollama(self, prompt: str) -> str:
        import ollama

        host = f"http://{self.config.ollama_host}:{self.config.ollama_port}"
        client = ollama.Client(host=host)
        response = client.chat(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            think=self.config.thinking
        )
        return response["message"]["content"].strip()

    def _parse_response(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("No JSON object found in model output: %.200s", text)
            return {"is_error": False, "error_description": None, "recommended_action": None}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model output: %.200s", text)
            return {"is_error": False, "error_description": None, "recommended_action": None}

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        batch_text = "\n".join(f"[{entry.index}] {entry.raw_text}" for entry in batch.entries)
        prompt = PROMPT_TEMPLATE.format(raw_text=batch_text)
        start = time.perf_counter()

        if self.config.backend == "ollama":
            raw_output = self._call_ollama(prompt)
        else:
            raw_output = self._call_hf(prompt)

        elapsed = time.perf_counter() - start
        logger.info("LLM batch #%d analyzed in %.3fs", batch.batch_id, elapsed)

        data = self._parse_response(raw_output)
        error_found = bool(data.get("is_error", False))
        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=error_found,
            error_count=1 if error_found else 0,
        )
