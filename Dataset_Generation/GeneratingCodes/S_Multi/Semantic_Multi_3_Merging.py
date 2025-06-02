import json
import ast
import random
from datetime import date, timedelta
from pathlib import Path
import os


random.seed(42)

import json 

def merging_dataset(base_path):  
    # Load the structured data
    structured_data_path = base_path + "Dataset_Helping/S_Multi/S_Multi_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/S_Multi/S_Multi_Structured_Generated_conversation.jsonl"

    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]

    print(f"Loaded {len(generated_data)} records from generated data file")

    dataset = []

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
            user_response = generated_data[int(i*30 + j)].replace(f"{user}:", "$$$$").replace(f"{user} ", "").replace("$$$$", f"{user}:")
            hour = random.randint(8, 17)
            minute = random.sample(range(0, 60),1)

    #         if user_response == '-':
    #             raise Exception("user_response is '-'")
    #         dataset.append({
    #             "user_ID": i,
    #             "topic": topic,
    #             "forum_question": forum_question,
    #             "message_date": message_date,
    #             "user": user,
    #             "context": f"{message_date} {hour:02d}:{minute[0]:02d}, {user}: {user_response}",
    #             "question": posts[j]['question'],
    #             "answer": posts[j]['answer']
    #         })

    # with open(base_path + 'Data/S_Multi.jsonl', 'w', encoding='utf-8') as f:
    #     for item in dataset:
    #         json.dump(item, f, ensure_ascii=False)
    #         f.write('\n')
    # print(f"S_Multi: {len(dataset)}, stored in {base_path}/Data/A_Multi.jsonl")

if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)