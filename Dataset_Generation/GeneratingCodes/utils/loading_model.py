import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm.auto import tqdm


class ModelLoader:
    def __init__(self, model_name, num_gpus):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.model = None
        self.backend = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.processor = None

        
        try:
            self.model = LLM(
                model=self.model_name,
                download_dir=os.environ.get("HF_HOME", None),
                tensor_parallel_size=self.num_gpus,
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                distributed_executor_backend="ray",
                enable_prefix_caching=True,
                max_model_len=4096,
            )
            self.backend = "vllm"
            print("Model loaded successfully using vLLM.")
        except Exception as e:
            if "Bfloat16" in str(e):
                print("Bfloat16 is not supported on this GPU. Retrying with dtype set to 'half'.")
                self.model = LLM(
                    model=self.model_name,
                    download_dir=os.environ.get("HF_HOME", None),
                    tensor_parallel_size=self.num_gpus,
                    gpu_memory_utilization=0.95,
                    trust_remote_code=True,
                    distributed_executor_backend="ray",
                    enable_prefix_caching=True,
                    dtype="half",
                    max_model_len=4096,
                )
                self.backend = "vllm"
                print("Model loaded successfully with dtype='half' using vLLM.")
            else:
                print(f"Error loading model: {e}")
                raise


    def generate(self, prompts, temperature=1.0, max_tokens=4096, batch_size=16):
        
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.model.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=True)
        return [output.outputs[0].text for output in outputs]
        