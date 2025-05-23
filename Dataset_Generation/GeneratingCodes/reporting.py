import os
from transformers import GPT2Tokenizer
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

def print_available_gpus():
    """Print information about available CUDA GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\nFound {gpu_count} CUDA GPU(s):")
            for i in range(gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\nNo CUDA GPUs available")
    except ImportError:
        print("\nPyTorch not installed - cannot check GPU availability")

def count_average_tokens():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Get path to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
    
    # Track totals for each file
    file_stats = {}
    
    print( os.listdir(data_dir))
    # Process each file in the data directory
    for filename in os.listdir(data_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        if filename.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        total_tokens = 0
        num_examples = 0
            
        for example in data:
            if 'context' in example:
                tokens = tokenizer.encode(example['context'])
                total_tokens += len(tokens)
                num_examples += 1
        
        if num_examples > 0:
            avg_tokens = total_tokens / num_examples
            file_stats[filename] = {
                'total_tokens': total_tokens,
                'avg_tokens': avg_tokens,
                'num_examples': num_examples
            }
    
    # Print results
    print("\nToken statistics for each file:")
    print("-" * 50)
    for filename, stats in file_stats.items():
        print(f"{filename}:")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average tokens: {stats['avg_tokens']:.1f}")
        print(f"  Number of examples: {stats['num_examples']:,}")
        print()

if __name__ == "__main__":
    print_available_gpus()
    count_average_tokens()



'''
Token statistics for each file:
--------------------------------------------------
S_Uni.jsonl:
  Total tokens: 579,556
  Average tokens: 386.4
  Number of examples: 1,500

A_Uni.jsonl:
  Total tokens: 559,141
  Average tokens: 372.8
  Number of examples: 1,500

S_Multi.jsonl:
  Total tokens: 71,329
  Average tokens: 142.7
  Number of examples: 500

T_Multi.jsonl:
  Total tokens: 65,608
  Average tokens: 131.2
  Number of examples: 500

T_Uni.jsonl:
  Total tokens: 537,144
  Average tokens: 358.1
  Number of examples: 1,500

A_Multi.jsonl:
  Total tokens: 73,879
  Average tokens: 147.8
  Number of examples: 500

'''