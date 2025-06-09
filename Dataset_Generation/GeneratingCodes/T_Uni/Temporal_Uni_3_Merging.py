import random
from datetime import timedelta, date, datetime
import ast
from pathlib import Path
import json
import os
# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------


random.seed(42)

def merging_dataset(base_path):

    # Load the structured data
    structured_data_path = base_path + "Dataset_Helping/T_Uni/T_Uni_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/T_Uni/T_Uni_Structured_Generated_conversation.jsonl"

    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]

    print(f"Loaded {len(generated_data)} records from generated data file")

    dataset = []


    x = 0
    for i in range(len(structured_data)):
        
        user_ID = i
        user = structured_data[i]['user_1']
        schedule_info_list = structured_data[i]['schedule']

        for j in range(len(schedule_info_list)):
            user_2 = schedule_info_list[j]['user_2']
            
            assert i*30 + j == x
            x += 1

            conversation = generated_data[int(i*30 + j)].replace(f"{user['name']}: {user_2['name']}:", f"{user_2['name']}:").replace(f"{user_2['name']}: {user['name']}:", f"{user['name']}:").replace(f"{user_2['name']}: {user_2['name']}:", f"{user_2['name']}:").replace(f"{user['name']}: {user['name']}:", f"{user['name']}:")



            if conversation == '-':
                print(conversation, i*30 + j)
                raise Exception("conversation is '-'")
            work_date = datetime.strptime(schedule_info_list[j]['question_time'][0], "%Y-%m-%d").strftime("%B %d, %Y")
            work_hour = schedule_info_list[j]['question_time'][1]
            question = f"What was {user} scheduled to be doing at {work_hour}:00 on {work_date}?"
            dataset.append({
                "user_ID": user_ID,
                "user": user['name'],
                "user_2": user_2['name'],
                "context": conversation.encode('utf-8').decode('unicode_escape'),
                "extra_info": {k:v for k,v in schedule_info_list[j].items() if k in ['activity_type', 'days', 'hours']},
                "question": question,
                "answer": schedule_info_list[j]['work']
            })

    with open(base_path + '/Data/T_Uni.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"T_Uni: {len(dataset)}, stored in {base_path}/Data/T_Uni.jsonl")
if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)