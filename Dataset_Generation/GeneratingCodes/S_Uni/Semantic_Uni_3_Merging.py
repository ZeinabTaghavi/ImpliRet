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
    structured_data_path = base_path + "Dataset_Helping/S_Uni/S_Uni_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/S_Uni/S_Uni_Structured_Generated_conversation.jsonl"

    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]

    print(f"Loaded {len(generated_data)} records from generated data file")

    dataset = []

    # Generate a random date in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    x = 0
    for i in range(len(structured_data)):

        user_ID = i
        user = structured_data[i]['user_1']
        trip_info_list = structured_data[i]['trip_info']


        for j in range(len(trip_info_list)):
            user_2 = trip_info_list[j]['user_2']
            if i*30 + j != x:
                print(i*30 + j, x)
            assert i*30 + j == x
            x += 1
            conversation = generated_data[int(i*30 + j)]

            conversation = [conv for conv in conversation.split('\n') if conv != '']
            
            # assert len(conversation) == 10, (conversation)

            random_days = random.randint(0, (end_date - start_date).days)
            message_date = start_date + timedelta(days=random_days)
            message_date = message_date.strftime("%Y-%m-%d")
            hour = random.randint(8, 17)
            minute = sorted(random.sample(range(0, 60), 10))


            # user_response = generated_data[int(i*30 + j)]
            if conversation == '-':
                print(conversation, i*30 + j)
                raise Exception("conversation is '-'")

            conversations = [f"{message_date} {hour:02d}:{minute[s]:02d}, {conversation[s].replace('user_2', user_2['name']).replace('User_2', user_2['name']).replace('user', user['name']).replace('User', user['name'])}" 
                              for s in range(len(conversation))]
            assert len(conversations) >= 9, f"{len(conversations)} != 10: {len(conversations)}"    

            question = f"What was {user['name']}'s reason for visiting {trip_info_list[j]['trip_country']}?"
            dataset.append({
                "user_ID": user_ID,
                "user": user['name'],
                "user_2": user_2['name'],
                "context": '\n'.join(conversations).replace(f"{user}: {user_2}:", f"{user_2}:").replace(f"{user_2}: {user}:", f"{user}:").replace(f"{user_2}: {user_2}:", f"{user_2}:").replace(f"{user}: {user}:", f"{user}:").encode('utf-8').decode('utf-8'),
                "extra_info": {k:v for k,v in trip_info_list[j].items() if k in ['type_of_location']},
                "question": question,
                "answer": trip_info_list[j]['trip_purpose']
            })

    with open(base_path + '/Data/F_Uni.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"F_Uni: {len(dataset)}, stored in {base_path}/Data/F_Uni.jsonl")

if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)

