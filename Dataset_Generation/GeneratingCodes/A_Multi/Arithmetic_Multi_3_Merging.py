import json
import ast
import random
from datetime import date, timedelta
from pathlib import Path
import os

random.seed(42)

import json 
import re

def generate_extra_samples(extra_samples, user_name, names_list, num_extra_samples):
    allowed_names = names_list.copy()
    reformed_extra_samples = []
    for es in extra_samples:
        month = random.randint(1, 12)
        # Get max days for the month, accounting for leap year in 2024
        if month in [4, 6, 9, 11]:
            max_days = 30
        elif month == 2:
            max_days = 29  # 2024 is a leap year
        else:
            max_days = 31
        random_date = date(2024, month, random.randint(1, max_days)).strftime('%Y-%m-%d')
        sample_name = random.choice(allowed_names)
        response = es['response']
        
        reformed_extra_samples.append(f"{random_date}, {sample_name}: {response}")
        if len(f"{random_date}, {sample_name}: {response}".split(".")) <= 4:
            print('--------------------------------')
            print(f"{random_date}, {sample_name}: {response}")
        
    assert len(reformed_extra_samples) == num_extra_samples, f"Expected {num_extra_samples} extra samples, got {len(reformed_extra_samples)}"
    return reformed_extra_samples

def merging_dataset(base_path):  
    # Load the structured data
    structured_data_path = base_path + "Dataset_Helping/A_Multi/A_Multi_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/A_Multi/A_Multi_Structured_Generated_conversation.jsonl"
    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]




    print(f'len(generated_data): {len(generated_data)}') 
    print(f"Loaded {len(generated_data)} records from generated data file")
    len(generated_data[0])

    dataset = []
    num_extra_samples = 149
    x = 0
    for i in range(len(structured_data)):
        
        topic = structured_data[i]['topic']
        forum_question = structured_data[i]['forum_question']
        posts = structured_data[i]['posts']

        for j in range(len(posts)):
            message_date = posts[j]['forum_post'][0]
            user = posts[j]['forum_post'][1]
            if i*30 + j != x:
                print(i*30 + j, x)
            assert i*30 + j == x
            x += 1

           
            user_response = generated_data[int(i*30 + j)].replace(f"{user['name']}:", "$$$$").replace(f"{user['name']} ", "").replace("$$$$", f"{user['name']}:")
            if user_response == '-':
                raise Exception("user_response is '-'")

            hour = random.randint(8, 17)
            minute = random.sample(range(0, 60),1)

            item_name = posts[j]['question'].split('model from which brand costs')[0].replace('what ', '').lower().strip()
            price = int(posts[j]['question'].split('that cost $')[1].replace('?', ''))
            price_format = str(price) if price < 1000 else f"{str(price)[:-3]},{str(price)[-3:]}"

            dataset.append({
                "user_ID": i,
                "topic": topic,
                "forum_question": forum_question,
                "message_date": f"{message_date} {hour:02d}:{minute[0]:02d}",
                "user": user['name'],
                "context": f"{message_date} {hour:02d}:{minute[0]:02d}, {user['name']}: {user_response}".encode('utf-8').decode('unicode_escape'),
                "question": f"What brand and model of {item_name} were priced at ${price_format}?",
                "answer": posts[j]['answer'],
            })

    with open(base_path + 'Data/A_Multi.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"A_Multi: {len(dataset)}, stored in {base_path}/Data/A_Multi.jsonl")

if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)