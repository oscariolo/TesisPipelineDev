import json
import logging
import re
import time
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult
from log_analysis.models.base import BaseModel
import ollama

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Determine if there is an error in this group of logs.Respond ONLY with valid JSON in this exact format, do not include any extra text or explanations:
{{
  "is_error": true or false,
  "error_description": "brief description of the error or null",
  "recommended_action": "what action to take or null"
}}

Log entry: {raw_text}"""

SYSTEM_PROMPT = {
        "role": "system",
        "content": """You are a log analysis assistant. You will be given a group of log entries. Your task is to determine if there is an error in this group of logs. Respond ONLY with valid JSON in this exact format, do not include any extra text or explanations.
{
    "is_error": true or false,
    "error_description": "brief description of the error or null",
    "recommended_action": "what action to take or null"
}""",
}


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
    token: str = None  # Optional API token for Ollama or other services
    bit_precision: Optional[str] = None  # e.g., "fp16", "int8", etc. for Hugging Face models
    tokenizer_name: Optional[str] = None  # e.g., base model repo when loading a GGUF model
    gguf_file: Optional[str] = None  # e.g., "Qwen3.6-27B-Q4_K_M.gguf" for GGUF repos



class GenerativeModel(BaseModel):
    def __init__(self, config: GenerativeConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._messages = []  # For chat history if applies
        self._context_window = config.contextWindow or self._getMaxContextWindow()
        self._token_usage = 0
        self.instanceModelClient()


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
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer_name = self.config.tokenizer_name or self.config.model_name
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            max_length = tokenizer.model_max_length
            return max_length
        return 4096

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

    def _call_hf(self) -> tuple[str, int]:
        inputs = self._tokenizer.apply_chat_template(
            self._messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(inputs, dict):
            inputs = {"input_ids": inputs}
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model.generate(**inputs, max_new_tokens=200)
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
        error_found = bool(data.get("is_error", False))

        logger.info("Current token usage: %d, context window: %d", self._token_usage, self._context_window)

        #tokenLimit mode: si supera un porcentaje de tokens se limpia historial para el siguiente batch (reinicio de contexto)
        if self.config.keepHistory == "tokenLimit" and self._token_usage >= self._context_window*self.config.tokenLimitPercentage:
            self._messages = []
            self._token_usage = 0

    
        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=error_found,
            model_name=self.config.model_name if hasattr(self.config, "model_name") else None,
            token_usage=self._token_usage,
        )
