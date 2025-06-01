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

        if self.model_name == "google/gemma-3-27b-it":
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name, device_map="auto"
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
            self.backend = "transformers"
            print("Gemma 3 model (multimodal) loaded with AutoProcessor.")
        else:
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
        if self.backend == "vllm":
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = self.model.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=True)
            return [output.outputs[0].text for output in outputs]
        elif self.backend == "transformers":
            outputs = []
            for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
                batch_prompts = prompts[batch_start: batch_start + batch_size]

                input_ids_list = []
                attention_list = []
                for chat in batch_prompts:
                    gemma_chat = []
                    for msg in chat:
                        if isinstance(msg["content"], str):
                            gemma_chat.append({
                                "role": msg["role"],
                                "content": [{"type": "text", "text": msg["content"]}]
                            })
                        else:
                            gemma_chat.append(msg)

                    single = self.processor.apply_chat_template(
                        gemma_chat,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    input_ids_list.append(single["input_ids"].squeeze(0))
                    attention_list.append(single["attention_mask"].squeeze(0))

                from torch.nn.utils.rnn import pad_sequence
                pad_id = self.processor.tokenizer.pad_token_id
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
                attention_mask = pad_sequence(attention_list, batch_first=True, padding_value=0)

                inputs = {
                    "input_ids": input_ids.to(self.model.device, dtype=torch.long),
                    "attention_mask": attention_mask.to(self.model.device, dtype=torch.long),
                }
                input_lens = [len(ids) for ids in input_ids_list]

                with torch.inference_mode():
                    generations = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                    )

                for i, gen in enumerate(generations):
                    continuation = gen[input_lens[i]:]
                    text = self.processor.decode(continuation, skip_special_tokens=True)
                    outputs.append(text)

                # free memory
                del input_ids, attention_mask, generations
                torch.cuda.empty_cache()

            return outputs
        else:
            raise ValueError("Model backend is not recognized. Please load the model first.")