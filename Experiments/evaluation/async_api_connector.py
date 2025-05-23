"""
Unified *async* API connector for the Metatag-Indexing experiments.

Supported providers
-------------------
openai          – OpenAI public Chat-Completions API  
azure-openai    – Azure OpenAI endpoint  
vllm            – Any OpenAI-compatible REST endpoint, e.g.
                  `python -m vllm.entrypoints.openai.api_server …`
local-vllm      – Load a HF model in-process with the vLLM Python API  
hf-local        – Load a HF model in-process with `transformers` only

All providers expose the same surface:

    .encode(text) / .decode(tok_ids) / .token_count(text)
    .generate_response(system_prompt, user_prompt, …) → dict
    .SYSTEM_PROMPT   (default = "You are a helpful assistant.")

The dict returned by *generate_response* always contains

    { response, prompt_tokens, completion_tokens,
      total_tokens, finish_reason }
"""

from __future__ import annotations
import time
import asyncio
from functools import cache
from typing import Any, Dict, List, Union
from openai import OpenAI

# ------------------------------------------------------------------ #
# third-party deps (all async-safe)                                   #
# ------------------------------------------------------------------ #
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

# vLLM is optional (only needed for local-vllm)
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


# ------------------------------------------------------------------ #
#                           APIConnector                             #
# ------------------------------------------------------------------ #
class APIConnector:
    """
    Initialise with **exactly** the keys from a model-config JSON:

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
        self.SYSTEM_PROMPT = "You are a helpful assistant."
    

        # 1) OpenAI or remote vLLM (same REST schema)
        if api_provider in {"openai", "vllm"}:
            self.api_provider = api_provider
            # self.api = AsyncOpenAI(
            #     api_key=api_key or "EMPTY",
            #     base_url=api_url if api_provider == "vllm" else None,
            #     max_retries=max_retries,
            #     timeout=timeout,
            # )
            self.api = OpenAI(api_key=api_key)
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

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

        # 3) Local vLLM in-process
        elif api_provider == "local-vllm":
            import torch

            if not _HAS_VLLM:
                raise ImportError("'vllm' is not installed (pip install vllm).")
            self.api_provider = api_provider
            self.num_gpus = torch.cuda.device_count()

            download_dir = kwargs.get("download_dir")
            if download_dir == "HF_HOME":
                import os
                download_dir = os.environ.get("HF_HOME", None)

            tp_size = int(kwargs.get("tensor_parallel_size", 1))
            gpu_util = float(kwargs.get("gpu_memory_utilization", 0.90))
            if tp_size > 1:
                print(f"[APIConnector] tensor_parallel_size={tp_size}. "
                      "Ensure you have enough GPU memory for all shards.")

            print(f"[APIConnector] Loading {model} via local vLLM "
                  f"(TP={tp_size}, util={gpu_util}, dir={download_dir})")
            backend = "ray" if kwargs.get("use_ray", False) else None
            max_len = kwargs.get("max_model_len", None)
            vllm_kwargs = dict(
                model=model,
                download_dir=download_dir,
                tensor_parallel_size=self.num_gpus,
                gpu_memory_utilization=gpu_util,
                # add prefix caching only if this vLLM version supports it
                # (parameter appeared in vLLM ≥ 0.4.2)
                )
            if hasattr(LLM, "enable_prefix_caching"):
                vllm_kwargs["enable_prefix_caching"] = True
            if backend:
                vllm_kwargs["distributed_executor_backend"] = backend
            if max_len:
                vllm_kwargs["max_model_len"] = max_len
            try:
                self.llm = LLM(**vllm_kwargs)
                print("Model loaded successfully.")
            except Exception as e:
                if "Bfloat16" in str(e):
                    print("Bfloat16 is not supported on this GPU. Retrying with dtype set to 'half'.")
                    self.llm = LLM(**vllm_kwargs, dtype="half")
                    print("Model loaded successfully with dtype='half'.")
                else:
                    print(f"Error loading model: {e}")
                    raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, use_fast=True, cache_dir=download_dir, trust_remote_code=True
            )
            # recompute tokenizer token‑lengths without special tokens
            self.encode = lambda txt: self.tokenizer(
                txt, add_special_tokens=False)["input_ids"]

        else:
            raise ValueError(f"Unsupported api_provider: {api_provider}")

    # ---------------- token helpers ---------------- #
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text,add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @cache
    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    # ---------------- core call ---------------- #
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        add_default_system_prompt: bool = True,
    ) -> Dict[str, Any]:
        """Return uniform dict regardless of provider."""

        # Build message list (OpenAI format)
        msgs: List[Dict[str, str]] = []
        if add_default_system_prompt and self.SYSTEM_PROMPT:
            msgs.append({"role": "system", "content": self.SYSTEM_PROMPT})
        elif system_prompt:
            msgs.append({"role": "system", "content": system_prompt})


        msgs.append({"role": "user", "content": user_prompt})

        # ---- A. remote OpenAI / vLLM / Azure ----
        if self.api_provider in {"openai", "vllm", "azure-openai"}:

            def _run():
                print('api call')
                time.sleep(0.01)
                return self.api.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **({"seed": 42} if self.api_provider == "openai" else {}),
                )

            completion = _run()
            choice = completion.choices[0]
            usage = completion.usage
            print(f"prompt_tokens: {completion.usage.prompt_tokens}")
            return {
                "response": completion.choices[0].message.content,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens if completion.usage.prompt_tokens_details else None,
            }

        else:
            raise ValueError(f"Unsupported api_provider: {self.api_provider}")
        





        