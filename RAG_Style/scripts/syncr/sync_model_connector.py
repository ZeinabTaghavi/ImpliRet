"""
sync_model_connector.py — Local-only vLLM connector for Llama 3.3 (70B).
Provides the same interface as async_api_connector.generate_response,
but runs synchronously via vLLM.chat().
"""

import os
import torch
from typing import Any, Dict, List, Union
from transformers import AutoTokenizer

# vLLM is optional (only needed for local-vllm)
try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

class ModelLoader:
    """
    Local-only vLLM connector (synchronous).
    Loads Llama-3.3-70B-Instruct via vLLM.chat.
    """
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        download_dir: str | None = "HF_HOME",
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        print("\n ----------- [STEP 1] Checking vLLM Installation -----------")
        if not _HAS_VLLM:
            raise ImportError("'vllm' is not installed (pip install vllm).")

        print("\n ----------- [STEP 2] Setting Up Model Configuration -----------")
        # Resolve cache dir
        download_dir = kwargs.get("download_dir")
        if download_dir == "HF_HOME":
            download_dir = os.environ.get("HF_HOME", None)

        # Determine parallelism
        tp_size = tensor_parallel_size or torch.cuda.device_count()
        if tp_size > 1:
            print(f"[ModelLoader] tensor_parallel_size={tp_size}. "
                  "Ensure you have enough GPU memory for all shards.")

        # Configure vLLM parameters
        gpu_util = float(kwargs.get("gpu_memory_utilization", 0.90))
        print(f"[ModelLoader] Loading {model} via local vLLM "
              f"(TP={tp_size}, util={gpu_util}, dir={download_dir})")

        # Build vLLM kwargs
        vllm_kwargs = {
            "model": model,
            "download_dir": download_dir,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len
        if hasattr(LLM, "enable_prefix_caching"):
            vllm_kwargs["enable_prefix_caching"] = True
        if "backend" in vllm_kwargs.keys():
            vllm_kwargs["distributed_executor_backend"] = vllm_kwargs["backend"]

        print("\n ----------- [STEP 3] Loading Model -----------")
        print('vllm_kwargs:')
        print(vllm_kwargs)
        print('-'*10)
        try:
            self.llm = LLM(**vllm_kwargs)
            print("Model loaded successfully.")
        except Exception as e:
            if "Bfloat16" in str(e):
                print("Bfloat16 not supported. Retrying with dtype='half'.")
                vllm_kwargs["dtype"] = "half"
                self.llm = LLM(**vllm_kwargs)
                print("Model loaded successfully with dtype='half'.")
            else:
                print(f"Error loading model: {e}")
                raise

        print("\n ----------- [STEP 4] Loading Tokenizer -----------")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            cache_dir=download_dir,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams
        self.SYSTEM_PROMPT = "You are a helpful assistant."
        print("Tokenizer loaded successfully.")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        add_default_system_prompt: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate responses for one or more prompts.
        
        Args:
            system_prompt: System message
            user_prompt: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            add_default_system_prompt: Whether to use default system prompt
            
        Returns:
            Single response dict or list of response dicts
        """
        # Handle single prompt
        if isinstance(user_prompt, str):
            user_prompt = [user_prompt]

        # Build message lists
        batch_msgs = []
        for up in user_prompt:
            msgs = []
            if add_default_system_prompt and self.SYSTEM_PROMPT:
                msgs.append({"role": "system", "content": self.SYSTEM_PROMPT})
            msgs.append({"role": "user", "content": up})
            batch_msgs.append(msgs)

        # Configure sampling
        sampling = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Generate responses
        outputs = self.llm.chat(
            messages=batch_msgs,
            sampling_params=sampling,
            use_tqdm=True
        )

        # Process results
        results = []
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text if hasattr(out.outputs[0], "text") else out.outputs[0]
            prompt_toks = self.token_count("\n".join(m["content"] for m in batch_msgs[i]))
            completion_toks = self.token_count(gen_text)
            results.append({
                "response": gen_text.strip(),
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
                "finish_reason": "local_vllm",
            })

        return results if len(results) > 1 else results[0]
