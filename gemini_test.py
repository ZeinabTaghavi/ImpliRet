# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-27b-it"
print(torch.cuda.device_count())

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    [messages, messages, messages], add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# Compute true lengths for each sequence (non‑padded tokens)
input_lens = (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# Decode results for each item in the batch
outputs = []
for i, gen in enumerate(generations):
    gen = gen[input_lens[i]:]  # strip the prompt part
    text = processor.decode(gen, skip_special_tokens=True)
    outputs.append(text)

for idx, out in enumerate(outputs, 1):
    print(f"Output {idx}: {out}\n")

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
