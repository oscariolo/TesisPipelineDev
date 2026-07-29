import json
import logging
import re
import time
from typing import Optional

from opentelemetry import metrics, trace
from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogEntry, AnalysisResult
from log_analysis.models.base import BaseModel

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("log-analysis.models")
meter = metrics.get_meter("log-analysis.models")

llm_input_tokens = meter.create_counter("llm.tokens.input", unit="tokens")
llm_output_tokens = meter.create_counter("llm.tokens.output", unit="tokens")
llm_call_duration = meter.create_histogram(
    "llm.call.duration", unit="s", description="LLM call duration"
)

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
    token: str = None  # Optional API token for Ollama or other services
    bit_precision: Optional[str] = None  # e.g., "fp16", "int8", etc. for Hugging Face models



class GenerativeModel(BaseModel):
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_hf(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading HF model %s on %s", self.config.model_name, self.config.hf_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, token=self.config.token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.hf_device,
            token=self.config.token
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

    def analyze(self, entry: LogEntry) -> AnalysisResult:
        with tracer.start_as_current_span("llm.analyze") as span:
            span.set_attributes({
                "log.index": entry.index,
                "llm.model": self.config.model_name,
                "llm.backend": self.config.backend,
                "llm.thinking": self.config.thinking,
            })

            prompt = PROMPT_TEMPLATE.format(raw_text=entry.raw_text)
            start = time.perf_counter()

            if self.config.backend == "ollama":
                raw_output = self._call_ollama(prompt)
            else:
                raw_output = self._call_hf(prompt)

            elapsed = time.perf_counter() - start
            llm_call_duration.record(elapsed)
            span.set_attribute("llm.duration_s", elapsed)

            input_tokens = len(prompt.split())
            output_tokens = len(raw_output.split())
            llm_input_tokens.add(input_tokens, {"llm.model": self.config.model_name, "llm.backend": self.config.backend})
            llm_output_tokens.add(output_tokens, {"llm.model": self.config.model_name, "llm.backend": self.config.backend})

            data = self._parse_response(raw_output)
            result = AnalysisResult(
                log_index=entry.index,
                is_error=data.get("is_error", False),
                error_description=data.get("error_description"),
                recommended_action=data.get("recommended_action"),
            )
            span.set_attribute("result.is_error", result.is_error)
            return result
