import json
import random
import ast
from datetime import date, timedelta, datetime

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

errored_list = [10]
errored_dict = {10 : '''I recently purchased a pair of Fendi sunglasses, and I can tell you that the 2016 model costs one thousand four hundred dollars, while the 2021 model costs twenty-five percent more, which is a significant difference. The Fendi 2021 model costs one thousand seven hundred fifty dollars, which is twenty-five percent more than the 2016 model. I chose the 2021 model because of its sleek design and superior UV protection. Overall, I think Fendi offers great quality and style, and the 2021 model is a worthwhile investment, which is why I ultimately decided to purchase it.''' }
random.seed(42)
def cleaning_conversation(conversation, num_utterance, names):
    conversation = conversation.replace('\\n', '\n')
    conversation = [conv.replace('user:', '').replace('User:', '').replace('user :', '').replace('User :', '').strip()
                     for conv in conversation.split("\n") if conv.strip() != '']
    conversation = [conv.replace('user_2', names[1]).replace('User_2', names[1]).replace('user', names[0]).replace('User', names[0]).strip()
                    for conv in conversation if conv.strip() != '']

    if len(conversation) > num_utterance:
        if names[0] in conversation or names[1] in conversation:
            new_conversation = []
            for i in range(len(conversation)-1):
                if names[0] in conversation[i] or names[1] in conversation[i]:
                    if names[0] in conversation[i+1] or names[1] in conversation[i+1]:
                        raise Exception(f"Conversation is not clean: {conversation}, {names}")
                    new_conversation.append(f"{conversation[i]}: {conversation[i+1]}")
            conversation = new_conversation
        # exit()
    assert len(conversation) <= num_utterance, f"Conversation does not have enough sentences: {conversation}, {len(conversation)}"
    # for conv in conversation:
        # assert names[0] in conv or names[1] in conv or 'Valerary' in conv or 'Elena' in conv or  "Aquallan" in conv or 'Zephka' in conv or 'Zeredla' in conv or 'Ziphra' in conv or 'Rafauele' in conv or 'Rafaiele' in conv or 'Lauder' in conv or 'Isaac' in conv or 'Isaca' in conv or 'Serilda' in conv or 'Isacad' in conv, f"User is not in conversation: {conv}, {names}"
    return conversation

def generate_extra_samples(extra_samples,  user, user_2, num_extra_samples):
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
        hour = random.randint(8, 19)
        minutes = sorted(random.sample(range(60), 10))


        response = es['response']
        conversation_utterance = cleaning_conversation(response, 10, [user, user_2])
        
        reformed_conversation_utterance = []
        for i in range(10):
            mentioned_user = conversation_utterance[i].split(":")[0]
            # assert mentioned_user == user or mentioned_user == user_2 or 'Elena' == mentioned_user or 'Valerary' == mentioned_user or 'Aquallan' == mentioned_user or 'Zephka' == mentioned_user or 'Zeredla' == mentioned_user or 'Ziphra' == mentioned_user or 'Rafauele' == mentioned_user or 'Rafaiele' == mentioned_user or 'Lauder' == mentioned_user or 'Isaac' == mentioned_user or 'Isaca' == mentioned_user or 'Serilda' == mentioned_user or 'Isacad' == mentioned_user, f"User is not the same: {user} and {user_2} , {mentioned_user}"
            random_minute = minutes[i]
            reformed_conversation_utterance.append(f"{random_date} {hour:02d}:{random_minute:02d}, {conversation_utterance[i]}")
        reformed_extra_samples.append(reformed_conversation_utterance)
        
    assert len(reformed_extra_samples) == num_extra_samples, f"Expected 149 extra samples, got {len(reformed_extra_samples)}"
    return reformed_extra_samples

def merging_dataset(base_path):

    # Load the structured data
    structured_data_path = base_path + "Dataset_Helping/A_Uni/A_Uni_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/A_Uni/A_Uni_Structured_Generated_conversation.jsonl"

    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]

    
    print(f"Loaded {len(generated_data)} records from generated data file")
  

    dataset = []
    num_extra_samples = 149

    # Generate a random date in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    x = 0
    for i in range(len(structured_data)):
        user_ID = i
        user = structured_data[i]['user_1']
        shopping_info_list = structured_data[i]['shopping_info']

        for j in range(len(shopping_info_list)):
            user_2 = shopping_info_list[j]['user_2']
            if i*30 + j != x:
                print(i*30 + j, x)
            assert i*30 + j == x
            x += 1

            conversation = generated_data[int(i*30 + j)]
            conversation = cleaning_conversation(conversation, 10, [user, user_2])
  

            random_days = random.randint(0, (end_date - start_date).days)
            message_date = start_date + timedelta(days=random_days)
            message_date = message_date.strftime("%Y-%m-%d")
            hour = random.randint(8, 17)
            minute = sorted(random.sample(range(0, 60), 10))

            conversation = [f"{message_date} {hour:02d}:{minute[s]:02d}, {conversation[s]}"
                              for s in range(len(conversation))]

            # user_response = generated_data[int(i*30 + j)]
            if conversation == '-':
                print(conversation, i*30 + j)
                raise Exception("conversation is '-'")


 
            price_format = str(shopping_info_list[j]['final_price']) if shopping_info_list[j]['final_price'] < 1000 else f"{str(shopping_info_list[j]['final_price'])[:-3]},{str(shopping_info_list[j]['final_price'])[-3:]}"
            question = f"What did {user} buy for ${price_format}?"
            dataset.append({
                "user_ID": user_ID,
                "user": user,
                "user_2": user_2,
                "context": '\n'.join(conversation).replace(f"{user}: {user_2}:", f"{user_2}:").replace(f"{user_2}: {user}:", f"{user}:").replace(f"{user_2}: {user_2}:", f"{user_2}:").replace(f"{user}: {user}:", f"{user}:"),
                "extra_info": {k:v for k,v in shopping_info_list[j].items() if k in ['shopping_type', 'item_to_buy', 'high_price_brand', 'low_price_brand']},
                "question": question,
                "answer": shopping_info_list[j]['final_shopping'],
            })
     

    with open(base_path + 'Data/A_Uni.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"A_Uni: {len(dataset)}, stored in {base_path}/Data/A_Uni.jsonl")

if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)
