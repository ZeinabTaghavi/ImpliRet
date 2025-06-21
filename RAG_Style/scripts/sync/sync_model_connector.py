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
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        # resolve cache dir
        print("\n ----------- [STEP 1] Checking vLLM Installation -----------")

        if not _HAS_VLLM:
            raise ImportError("'vllm' is not installed (pip install vllm).")

        print("\n ----------- [STEP 2] Setting Up Model Configuration -----------")
        download_dir = kwargs.get("download_dir")
        if download_dir == "HF_HOME":
            download_dir = os.environ.get("HF_HOME", None)
        # determine parallelism
        tp_size = tensor_parallel_size or torch.cuda.device_count()
        if tp_size > 1:
            print(f"[ModelLoader] tensor_parallel_size={tp_size}. "
                    "Ensure you have enough GPU memory for all shards.")

        gpu_util = float(kwargs.get("gpu_memory_utilization", 0.90))
        print(f"[ModelLoader] Loading {model} via local vLLM "
            f"(TP={tp_size}, util={gpu_util}, dir={download_dir})")
        # instantiate vLLM
        backend = "ray" if kwargs.get("use_ray", False) else None
        vllm_kwargs = dict(
            model=model,
            download_dir=download_dir,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len
        if hasattr(LLM, "enable_prefix_caching"):
            vllm_kwargs["enable_prefix_caching"] = True
        if backend:
            vllm_kwargs["distributed_executor_backend"] = backend
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len

        print("\n ----------- [STEP 3] Loading Model -----------")
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
                
        print("\n ----------- [STEP 4] Loading Tokenizer -----------")
        # tokenizer for token counting
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
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: List[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        add_default_system_prompt: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        print("\n ----------- [STEP 1] Preparing Messages -----------")
        # build batch of message lists
        batch_msgs = []
        for up in user_prompt:
            msgs = []
            if add_default_system_prompt and self.SYSTEM_PROMPT:
                msgs.append({"role": "system", "content": self.SYSTEM_PROMPT})
            msgs.append({"role": "user", "content": up})
            batch_msgs.append(msgs)

        print("\n ----------- [STEP 2] Setting Sampling Parameters -----------")
        # sampling params
        sampling = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        print("\n ----------- [STEP 3] Generating Responses -----------")
        # synchronous batched chat call
        outs = self.llm.chat(
            messages=batch_msgs,
            sampling_params=sampling,
            use_tqdm=True
        )

        print("\n ----------- [STEP 4] Processing Results -----------")
        # collect results
        results = []
        for i, inst in enumerate(outs):
            gen_out = inst.outputs[0]
            text = gen_out.text if hasattr(gen_out, "text") else gen_out
            # token counts
            prompt_toks = self.token_count("\n".join(m["content"] for m in batch_msgs[i]))
            completion_toks = self.token_count(text)
            results.append({
                "response": text.strip(),
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
                "finish_reason": "local_vllm",
            })

        # return single dict for single prompt, else list
        return results if len(results) > 1 else results[0]
