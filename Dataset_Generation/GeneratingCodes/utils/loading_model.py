import os
from vllm import LLM, SamplingParams

def load_model(model_name, num_gpus):
    try:
        llm = LLM(model=model_name,
            download_dir=os.environ["HF_HOME"],
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            distributed_executor_backend="ray",
            enable_prefix_caching=True,
            max_model_len=4096,
        )
        print("Model loaded successfully.")
    except Exception as e:
        if "Bfloat16" in str(e):
            print("Bfloat16 is not supported on this GPU. Retrying with dtype set to 'half'.")
            llm = LLM(model=model_name,
                download_dir=os.environ["HF_HOME"],
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                distributed_executor_backend="ray",
                enable_prefix_caching=True,
                dtype="half",
                max_model_len=4096,
            )
            print("Model loaded successfully with dtype='half'.")
        else:
            print(f"Error loading model: {e}")
            raise
    
    return llm