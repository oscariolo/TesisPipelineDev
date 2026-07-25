import json
import logging
import re
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogEntry, AnalysisResult
from log_analysis.models.base import BaseModel

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Analyze this server log entry. Respond ONLY with valid JSON in this exact format (no other text):
{{
  "is_error": true or false,
  "error_description": "brief description of the error or null",
  "recommended_action": "what action to take or null"
}}

Log entry: {raw_text}

JSON response:"""


class GenerativeConfig(PydanticBaseModel):
    model_name: str
    backend: str
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    hf_device: str = "cpu"
    thinking: bool = False 


class GenerativeModel(BaseModel):
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_hf(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HF model %s on %s", self.config.model_name, self.config.hf_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.hf_device,
        )

    def _call_hf(self, prompt: str) -> str:
        if self._model is None:
            self._load_hf()
        inputs = self._tokenizer(prompt, return_tensors="pt")
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

    def analyze(self, entry: LogEntry) -> AnalysisResult:
        prompt = PROMPT_TEMPLATE.format(raw_text=entry.raw_text)

        if self.config.backend == "ollama":
            raw_output = self._call_ollama(prompt)
        else:
            raw_output = self._call_hf(prompt)

        data = self._parse_response(raw_output)
        return AnalysisResult(
            log_index=entry.index,
            is_error=data.get("is_error", False),
            error_description=data.get("error_description"),
            recommended_action=data.get("recommended_action"),
        )
