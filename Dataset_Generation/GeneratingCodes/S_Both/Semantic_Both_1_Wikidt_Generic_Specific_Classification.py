import os
from huggingface_hub import login
from vllm import LLM, SamplingParams
import json
import argparse
import random
import re
import os
import torch
import ast

random.seed(42)

# Load the JSONL files
def load_jsonl(filename):
    try:
        with open(filename, 'r') as file:
            return [json.loads(line) for line in file]
    except:
        with open(filename, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]


def filter_generated_conversation_step_1(generated_conversation):

    conversation_list = []
    mistaken_conversation_idx = []
    for idx , text in enumerate(generated_conversation):
        # Find all matches; each match is a tuple of three strings
        matches = text.split('CLASSIFICATION:')[-1].strip()

        if ("generic" in matches.lower() and "specific" not in matches.lower()) or ("specific" in matches.lower() and "generic" not in matches.lower()):
            if "generic" in matches.lower():
                if "hint" in text.lower():
                    conversation_list.append("Generic - Hint")
                else:
                    conversation_list.append("Generic")
            else:
                if "hint" in text.lower():
                    conversation_list.append("Specific - Hint")
                else:
                    conversation_list.append("Specific")
        else:
            mistaken_conversation_idx.append(idx)
            conversation_list.extend('-')
            
    return mistaken_conversation_idx, conversation_list


# Generate the outputs
def generating_outputs(dataset, llm, prompt_1, output_filename):

    wikidt_list = []
    # step 1 is generating the conversation
    prompts = []
    for i in range(len(dataset)):
        wikidt_item = dataset[i]['itemLabel']
        conversation = [
                {
                    "role": "system",
                    "content": prompt_1
                },
                {
                    "role": "user",
                    "content": f"{wikidt_item}"
                }
            ]

        wikidt_list.append(wikidt_item)
        prompts.append(conversation)

    # 4. Generate the outputs at step 1
    print("Generating outputs in step 1...")
    print(len(prompts))
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    
    # Process in batches of 1012 samples
    batch_size = 4048
    llm_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(prompts)-1)//batch_size + 1}")
        
        outputs = llm.chat(messages=batch_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
        llm_outputs.extend([o.outputs[0].text for o in outputs])

    mistaken_conversation_idx = [i for i in range(len(prompts))]
    print("End of step 1")


    conversation_total_attempts = 2

    print(f"Number of mistaken conversations: {len(mistaken_conversation_idx)}")

    for attempt in range(conversation_total_attempts):
        print(f"1 - Conversation Generation: Attempt {attempt} of {conversation_total_attempts}")
        mistaken_conversation_idx, conversation_list = filter_generated_conversation_step_1(llm_outputs)
        if len(mistaken_conversation_idx) == 0:
            print("No mistaken conversations")
            break
        else:
            mistaken_prompts = [prompts[i] for i in mistaken_conversation_idx]
            sampling_params = SamplingParams(temperature=0, max_tokens=128)
            outputs = llm.chat(messages=mistaken_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
            re_generated_llm_outputs = [o.outputs[0].text for o in outputs]
            for i in range(len(mistaken_conversation_idx)):
                llm_outputs[mistaken_conversation_idx[i]] = re_generated_llm_outputs[i]

    mistaken_conversation_idx, conversation_list = filter_generated_conversation_step_1(llm_outputs)
    print(f"Number of mistaken conversations: {len(mistaken_conversation_idx)}")
    if len(mistaken_conversation_idx) == 0:
        for i in range(len(conversation_list)):
            dataset[i]['LLM_GS_Clf'] = conversation_list[i]
            return dataset
    return conversation_list
    

def main(num_gpus, model_name):
    # 1. Loading the model
    print("Loading the model...")
    try:
        ## Load the model
        llm = LLM(model=model_name,
            download_dir=os.environ["HF_HOME"],
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            distributed_executor_backend="ray",
            enable_prefix_caching=True,
            max_model_len=1024,  # Limit sequence length to a value that fits in GPU memory
        )
        print("Model loaded successfully.")
    except Exception as e:
        if "Bfloat16" in str(e):
            print("Bfloat16 is not supported on this GPU. Retrying with dtype set to 'half'.")
            # Retry by explicitly setting dtype to 'half'
            llm = LLM(model=model_name,
                download_dir=os.environ["HF_HOME"],
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                distributed_executor_backend="ray",
                enable_prefix_caching=True,
                dtype="half",
                max_model_len=1024,  # Limit sequence length to a value that fits in GPU memory
            )
            print("Model loaded successfully with dtype='half'.")
        else:
            print(f"Error loading model: {e}")
            raise

    
    prompt_1 = '''
You will be given string INPUT representing a location name (e.g. “former cinema Lowen-Kino”).

Your task is to classify INPUT as **Generic** or **Specific** and indicate whether its language lets you guess the country of origin (a “Hint”).

Definitions:

1. **Generic Location**  
   - A broad or common place type, not a unique landmark.  
   - Includes descriptive or placeholder names (e.g. "Theatre Downtown", "House 2345243", "Thomas Office").  
   - If you're uncertain whether it denotes one well-known site, treat it as Generic.  
   - Common names shared by many places (e.g. "Santa Maria") are Generic unless it clearly refers to one famous site.

2. **Specific Location**  
   - A clearly named, unique landmark or building (e.g. “Buckingham Palace,” “Old Cadet Chapel”).  
   - Only Specific if it unambiguously denotes a single, recognized place.

Additional requirement:

3. **Language Hint**  
   - If the name's wording (e.g. German, French, Japanese terms) provides a clue to its country, append **“‑ Hint”** after the classification.  
   - Otherwise, don't add a hint.

Instructions:

1. Read the INPUT carefully.  
2. Decide **Generic** or **Specific** per the definitions above.  
3. Decide if the item's language suggests its country of origin.  
4. Output exactly one line in this format: 

[Brief justification] CLASSIFICATION: [Generic or Specific][- Hint]

- Keep the explanation to 1-2 sentences.  
- Append `- Hint` only if the name's language clearly points to a country.

Example outputs:

- `Looks like a general venue name, not a unique landmark with spanish words. CLASSIFICATION: GENERIC - Hint`  
- `Name uses German words "Das Kirche", so you can guess it's in Germany. CLASSIFICATION: SPECIFIC - Hint`  
- `It's a generic descriptor in English with no country clue. CLASSIFICATION: GENERIC`  

Write your single-line answer now:
    '''

    # 2. Loading the dataset
    dataset = load_jsonl(f'./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_2_low_scored_records.jsonl')

    output_filename = "./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_3_LLM_Generic_Specific_Classification.jsonl"

    # 3. Generating the outputs
    llm_outputs = generating_outputs(dataset, llm, prompt_1, output_filename)
        
    # Save LLM outputs 
    with open(output_filename, 'w') as f:
        for output in llm_outputs:
            json.dump(output, f)
            f.write('\n')
    print(f"LLM outputs saved to {output_filename}")


    # Ensure that you explicitly destroy the process group at the end of your script. 
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    print("Process group destroyed.")


if __name__ == '__main__':

    # meta-llama/Meta-Llama-3-8B-Instruct - meta-llama/Llama-3.3-70B-Instruct
    parser = argparse.ArgumentParser(description="First Experiment:")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="LLM Name.")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    model_name = args.model_name

    print("----- Experiment Setup ----")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"Number of GPUs: {num_gpus}")
    print('---------------------------')

    # Set HF_HOME environment variable
    login(token=os.environ.get("HF_TOKEN"))
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token")

    main(num_gpus, model_name)