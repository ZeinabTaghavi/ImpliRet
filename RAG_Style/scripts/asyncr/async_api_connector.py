# --------------------------------------------------------------------------- #
#                           Unified Async API Connector                         #
# --------------------------------------------------------------------------- #
"""
Unified async API connector for the Metatag-Indexing experiments.

Supported providers:
- openai: OpenAI public Chat-Completions API
- azure-openai: Azure OpenAI endpoint  
- vllm: Any OpenAI-compatible REST endpoint (e.g. vLLM API server)

All providers expose the same interface:
- .encode(text) / .decode(tok_ids) / .token_count(text)
- .generate_response(system_prompt, user_prompt, ...) -> dict
- .SYSTEM_PROMPT (default = "You are a helpful assistant.")

The dict returned by generate_response always contains:
- response: str
- prompt_tokens: int 
- completion_tokens: int
- total_tokens: int
- finish_reason: str
"""

from __future__ import annotations
import time
import asyncio
from functools import cache
from typing import Any, Dict, List, Union

# --------------------------------------------------------------------------- #
#                           Third-party Dependencies                            #
# --------------------------------------------------------------------------- #
from tenacity import (
    retry,
    wait_random,
    stop_after_delay,
    retry_if_exception_type,
)
import tiktoken
from openai import AsyncOpenAI, AsyncAzureOpenAI
try:
    from openai import RateLimitError  # new client
except ImportError:
    from openai.error import RateLimitError  # old client
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    pipeline,
)
import httpx


# --------------------------------------------------------------------------- #
#                              Main API Connector                               #
# --------------------------------------------------------------------------- #
class APIConnector:
    """
    Main API connector class for model inference.
    
    Initialize with model config JSON containing:
    {
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "api_provider": "local-vllm",
      "download_dir": "HF_HOME",
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.9,
      "max_tokens": 256,
      "temperature": 0.0,
      "top_p": 1.0,
      ...
    }
    """

    def __init__(
        self,
        api_provider: str,
        model: str,
        api_key: str = "",
        api_url: str = "",
        max_retries: int = 3,
        timeout: int = 600,
        azure_api_version: str | None = None,
        **kwargs: Any,
    ) -> None:

        self.model = model
        self.api_provider = api_provider
        self.SYSTEM_PROMPT = "You are a helpful assistant."

        # 1) OpenAI or remote vLLM (same REST schema)
        if api_provider in {"openai", "vllm"}:
            self.api = AsyncOpenAI(
                api_key=api_key or "EMPTY",
                base_url=api_url if api_provider == "vllm" else None,
                max_retries=max_retries,
                timeout=timeout,
            )
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except Exception:
                self.tokenizer = tiktoken.get_encoding("o200k_base")

        # 2) Azure OpenAI
        elif api_provider == "azure-openai":
            self.api_provider = api_provider
            self.api = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=azure_api_version or "2023-09-01-preview",
                max_retries=max_retries,
                timeout=timeout,
            )
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

        else:
            raise ValueError(f"Unsupported api_provider: {api_provider}")

    # --------------------------------------------------------------------------- #
    #                              Token Helpers                                    #
    # --------------------------------------------------------------------------- #
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @cache
    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    # --------------------------------------------------------------------------- #
    #                              Core API Call                                    #
    # --------------------------------------------------------------------------- #
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        add_default_system_prompt: bool = True,
    ) -> Dict[str, Any]:
        """Return a uniform dict regardless of provider (ASYNC version)."""

        # Build message list (OpenAI format)
        # Assemble message list in OpenAI‑style schema
        messages: List[Dict[str, str]] = []
        if add_default_system_prompt and self.SYSTEM_PROMPT:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})


        # Add user messages (accept list‑of‑turns or a single string)
        if isinstance(user_prompt, list):
            for up in user_prompt:
                messages.append({"role": "user", "content": up})
        else:
            messages.append({"role": "user", "content": user_prompt})

        # Helper with automatic retries for rate‑limits & transient errors
        @retry(
            reraise=True,
            wait=wait_random(min=1, max=20),
            stop=stop_after_delay(300),
            retry=retry_if_exception_type(RateLimitError),
        )
        async def _call_openai():
            return await self.api.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                # deterministic seed for reproducibility (OpenAI endpoints only)
                **({"seed": 43} if self.api_provider == "openai" else {}),
            )
        
        # ---- A. remote OpenAI / vLLM / Azure ----
        if self.api_provider in {"openai", "vllm", "azure-openai"}:

            completion = await _call_openai()
            choice = completion.choices[0]
            usage = completion.usage
            return {
                "response": choice.message.content,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "finish_reason": choice.finish_reason,
                "cached_tokens": (
                    usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details
                    else None
                ),
            }

        else:
            raise ValueError(f"Unsupported api_provider: {self.api_provider}")        




