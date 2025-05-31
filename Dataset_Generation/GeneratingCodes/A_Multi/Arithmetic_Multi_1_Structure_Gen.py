import json
import ast
import random
from datetime import date, timedelta
from pathlib import Path
import os


random.seed(42)

def save_jsonl(data, filename):
    """Save data to a JSONL file."""
    with open(filename, "w+", encoding="utf-8") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def find_valid_list_prices(num_items=30, max_tries=10, max_price=3000):
    """
    Generate valid list prices for items.
    
    Args:
        num_items: Number of price pairs to generate
        max_tries: Maximum attempts to generate valid prices
        max_price: Maximum allowed price
        
    Returns:
        Two lists containing paired prices
    """

    for _ in range(20):
        base_price_1 = list(range(0,num_items))
        random.shuffle(base_price_1)
        prices = []
        all_prices = []
        for i in base_price_1:
            k_min = 100*(i+1) - 50
            k_max = 100*(i+1) + 50
            
            for k in range(k_min, k_max, 10):
                
                if k > (max_price/2):
                    found = False
                    for _ in range(600):
                        l = (random.randint(100,  k) // 10) * 10
                        k_div_l = k / l
                        l_div_k = l / k
                        if (round(k_div_l, 2) == k / l) and (round(l_div_k, 2) == l / k) and (k_div_l < 3) and (l_div_k < 3) and (l != k) and (l not in all_prices) and (k not in all_prices):
                            if ((k,l) not in prices) and ((l,k) not in prices) and (l not in all_prices) and (k not in all_prices):
                                all_prices.append(k)
                                all_prices.append(l)
                                prices.append((k,l))
                                found = True
                                break
                    if found == True:
                        break
                if k < (max_price/2):
                    found = False
                    for _ in range(600):
                        l = (random.randint(k, max_price) // 10) * 10
                        l_div_k = l / k
                        k_div_l = k / l
                        if (round(l_div_k, 2) ==  l / k) and (round(k_div_l, 2) == k / l) and (l_div_k < 3) and (k_div_l < 3) and (l != k) and (l not in all_prices) and (k not in all_prices):
                            if ((k,l) not in prices) and ((l,k) not in prices) and (l not in all_prices) and (k not in all_prices):
                                all_prices.append(k)
                                all_prices.append(l)
                                prices.append((k,l))
                                found = True
                                break
                    if found == True:
                        break

        
        
        # Assert no duplicates between l_list and k_list
        if len(all_prices) != len(set(all_prices)) or (len(all_prices) != 2 * num_items):
            raise Exception(f"Found duplicate value in all_prices")


        if len(prices) == num_items:
            list_price_1 = [item_1 for item_1, item_2 in prices]
            list_price_2 = [item_2 for item_1, item_2 in prices]
            assert (len(list(set(list_price_1 + list_price_2))) == len(list_price_1 + list_price_2))
            assert (len(list(set(list_price_1 + list_price_2))) == int(2*num_items))
            return list_price_1, list_price_2 
        
    raise Exception("Failed to generate valid list prices")


def generate_message_prices(list_price_1 , list_price_2, names, brands, topic, question, year=2024):
    """
    Generate forum messages with price comparisons.
    
    Args:
        list_price_1: First list of prices
        list_price_2: Second list of prices 
        names: List of user names
        brands: List of brand names
        topic: Forum topic
        question: Question template
        year: Year for message dates
        
    Returns:
        List of message dictionaries with price comparisons
    """
    message_prices = []
    random.shuffle(brands)
    item_to_buy = topic
    
    for idx in range(len(list_price_1)):
        brand = brands[idx]
        price = list_price_1[idx]
        low_price_model = random.randint(2013,2022)
        high_price_model = random.randint(low_price_model+1,2024)
        
        # Handle case where first price is higher
        if list_price_1[idx] > list_price_2[idx]:
            mul = list_price_1[idx] / list_price_2[idx]
            model = high_price_model
            
            if mul < 2:
                if mul < 1:
                    raise Exception("mul is less than 1")
                mul = ((list_price_1[idx] - list_price_2[idx]) / list_price_2[idx]) * 100
                mul = round(mul, 2)
                if int(mul) == mul:
                    mul = int(mul)

                bought = [f"{item_to_buy} from {brand} model {low_price_model}: {list_price_2[idx]} dollars",
                         f"{brand}, model {high_price_model}: {mul} percents more expensive than model {low_price_model}",
                         f"model {high_price_model} was purchased"]
            else:
                mul = round(mul, 2)
                if int(mul) == mul:
                    mul = int(mul)
                bought = [f"{item_to_buy} from {brand} model {low_price_model}: {list_price_2[idx]} dollars",
                         f"{brand}, model {high_price_model}: {mul} times more expensive than model {low_price_model}",
                         f"model {high_price_model} was purchased"]
                         
            high_price_brand = ("high_priced_brand", list_price_1[idx])
            low_price_brand = ("low_priced_brand", list_price_2[idx])
            
        # Handle case where second price is higher
        else:
            mul = list_price_2[idx] / list_price_1[idx]
            model = low_price_model
            
            if mul < 2:
                if mul < 1:
                    raise Exception("mul is less than 1")
                mul = ((list_price_2[idx] - list_price_1[idx]) / list_price_2[idx]) * 100
                mul = round(mul, 2)
                if int(mul) == mul:
                    mul = int(mul)
                bought = [f"{item_to_buy} from {brand} model {high_price_model}: {list_price_2[idx]} dollars",
                         f"{brand}, model {low_price_model}: {mul} percents less expensive than model {high_price_model}",
                         f"model {low_price_model} was purchased"]
            else:
                mul = round(mul, 2)
                if int(mul) == mul:
                    mul = int(mul)
                bought = [f"{item_to_buy} from {brand} model {high_price_model}: {list_price_2[idx]} dollars",
                         f"{brand}, model {high_price_model}: {mul} times more expensive than model {low_price_model}",
                         f"model {low_price_model} was purchased"]
                
            high_price_brand = ("high_priced_brand", list_price_1[idx])
            low_price_brand = ("low_priced_brand", list_price_2[idx])

        random_date = f"{year}-{random.randint(1,12)}-{random.randint(1,28)}"
        message_prices.append({
            "forum_post": (random_date, names[idx], bought),
            "question": question.format(price=price),
            "answer": (brand, model)
        })
        
    return message_prices


def generate_dataset(num_items=20, max_price=3000):
    """
    Generate the complete dataset with forum topics and price comparisons.
    
    Args:
        num_items: Number of items per topic
        max_price: Maximum price allowed
    """

    # Load user names
    names_list = []
    with open("./Dataset_Generation/Dataset_Helping/names.jsonl", "r") as file:
        for line in file:
            names_list.append(json.loads(line.strip()))
   

    # Load forum topics
    forum_topics = []
    with open("./Dataset_Generation/Dataset_Helping/A_Multi/A_Multi_forum.jsonl", "r") as file:
        for line in file:
            forum_topic = json.loads(line.strip())
            forum_topics.append(forum_topic)

    # Generate dataset records for each topic
    dataset = []
    shuffled_names_list = names_list.copy()
    
    for topic in forum_topics:
        dataset_row = {
            'topic': topic["topic"],
            'forum_question': topic["forum_question"],
        }

        # Generate prices and messages
        list_price_1, list_price_2 = find_valid_list_prices(num_items, max_price)
        topic_users = random.sample(shuffled_names_list, 30)
     
        message_dates = generate_message_prices(list_price_1, list_price_2, topic_users, topic['items'], topic["topic"], topic['question'])
        dataset_row["posts"] = message_dates
        dataset.append(dataset_row)
  

    # Save dataset
    output_dir = Path("./Dataset_Generation/Dataset_Helping/A_Multi")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"A_Multi_Structured.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Step1: Arithmetic Multi Dataset Structure saved to {output_file}") 


if __name__ == "__main__":
    generate_dataset()