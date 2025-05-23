import json
import argparse
import ast
import argparse
import random
from datetime import date, timedelta
from pathlib import Path

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)


def generate_tuples_for_category(category_data, shopping_type):
    """
    Generate tuples of shopping information for a given category.
    
    Args:
        category_data: Dictionary containing high_priced, low_priced and items_to_buy lists
        shopping_type: Type of shopping category
        
    Returns:
        List of 5 tuples containing (shopping_type, item_to_buy, high_priced, low_priced)
    """
    indices = list(range(len(category_data["high_priced"])))
    random.shuffle(indices)
    tuples_list = []
    for idx in indices:
        item_to_buy = random.choice(category_data["items_to_buy"])
        high = category_data["high_priced"][idx]
        low = category_data["low_priced"][idx]
        tup = (shopping_type, item_to_buy, high, low)
        tuples_list.append(tup)
    return tuples_list

def find_valid_list_prices(num_items=30, max_tries=10, max_price=3000):
    """
    Generate valid pairs of prices that satisfy ratio constraints.
    
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


def generate_dataset(num_users = 50, num_items = 30, max_price = 3000):
    """
    Generate the full dataset with shopping information for multiple users.
    
    Args:
        num_users: Number of users to generate data for
        num_items: Number of shopping items per user
        max_price: Maximum price allowed for items
        
    Returns:
        None, saves dataset to file
    """
    # ------------------------------------------------------------
    # Load input data
    # ------------------------------------------------------------

    with open("./Dataset_Generation/Dataset_Helping/names.txt", "r") as file:
        names = ' '.join([line.strip() for line in file.readlines()])
        names_list = list(ast.literal_eval(names))

    with open("./Dataset_Generation/Dataset_Helping/A_Uni/A_Uni_Base_JSON.json", "r") as f:
        shopping_data = json.load(f)

    # ------------------------------------------------------------
    # Generate user data
    # ------------------------------------------------------------

    users = random.sample(names_list, num_users)

    category_tuples = {}
    for shopping_type, data in shopping_data.items():
        category_tuples[shopping_type] = generate_tuples_for_category(data, shopping_type)

    user_dataset = {}
    for user in users:
        user_dataset[user] = {"user_1": user, "shopping_info": []}
        all_tuples = []
        for shopping_type in category_tuples:
            all_tuples.extend(category_tuples[shopping_type])
        selected_tuples = random.sample(all_tuples, num_items)
        list_price_1 , list_price_2 =  find_valid_list_prices(num_items, max_price)
        
        # ------------------------------------------------------------
        # Generate shopping information for each user
        # ------------------------------------------------------------
        
        for idx, tup in enumerate(selected_tuples):
            shopping_info = {}
            shopping_type, item_to_buy, high_brand, low_brand = tup

            if list_price_1[idx] > list_price_2[idx]:
                mul = list_price_1[idx] / list_price_2[idx]
                item_was_bought = f"{item_to_buy} - {high_brand}"
                if mul < 2:
                    if mul < 1:
                        raise Exception("mul is less than 1")
                    mul = ((list_price_1[idx] - list_price_2[idx]) / list_price_2[idx]) * 100
                    mul = round(mul, 2)
                    if int(mul) == mul:
                        mul = int(mul)
                    bought = (f"{item_to_buy} from {low_brand} is {list_price_2[idx]}",
                            f"{item_to_buy} from {high_brand} is {mul} percent more expensive than {item_to_buy} from {low_brand}",
                            f"bought {item_to_buy} from {high_brand}")
                else:
                    mul = round(mul, 2)
                    if int(mul) == mul:
                        mul = int(mul)
                    bought = (f"{item_to_buy} from {low_brand} is {list_price_2[idx]}",
                            f"{item_to_buy} from {high_brand} is {mul} times more expensive than {item_to_buy} from {low_brand}",
                            f"bought{item_to_buy} from {high_brand}")
                high_price_brand = (high_brand, list_price_1[idx])
                low_price_brand = (low_brand, list_price_2[idx])
                
            else:
                mul = list_price_2[idx] / list_price_1[idx]
                item_was_bought = f"{item_to_buy} - {low_brand}"
                if mul < 2:
                    if mul < 1:
                        raise Exception("mul is less than 1")
                    mul = ((list_price_2[idx] - list_price_1[idx]) / list_price_2[idx]) * 100
                    mul = round(mul, 2)
                    if int(mul) == mul:
                        mul = int(mul)
                    bought = (f"{item_to_buy} from {high_brand} is {list_price_2[idx]}",
                            f"{item_to_buy} from {low_brand} is {mul} percent less expensive than {item_to_buy} from {high_brand}",
                            f"bought {item_to_buy} from {low_brand}")
                else:
                    mul = round(mul, 2)
                    if int(mul) == mul:
                        mul = int(mul)
                    bought = (f"{item_to_buy} from {high_brand} is {list_price_2[idx]}",
                            f"{item_to_buy} from {high_brand}is {mul} times more expensive than {item_to_buy} from {low_brand} ",
                            f"bought {item_to_buy} from {low_brand}")
                    
                high_price_brand = (low_brand, list_price_2[idx])
                low_price_brand = (high_brand, list_price_1[idx])

            user_2 = random.choice([u for u in names_list if u != user])
            shopping_info = {"user_2": user_2,
                            "shopping_type": shopping_type,
                            "item_to_buy": item_to_buy,
                            "high_price_brand": high_price_brand,
                            "low_price_brand": low_price_brand,
                            "bought": bought,
                            "final_price": list_price_1[idx],
                            "final_shopping": item_was_bought}
        
            user_dataset[user]["shopping_info"].append(shopping_info)

    # ------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------

    output_file = './Dataset_Generation/Dataset_Helping/A_Uni/A_Uni_Structured.jsonl'
    with open(output_file, 'w') as f:
        for user, data in user_dataset.items():
            line = {'user_1': user, 'shopping_info': data['shopping_info']}
            f.write(json.dumps(line) + '\n')

    print(f"Step1: Arithmetic Uni Dataset Structure saved to {output_file}")


if __name__ == "__main__":
    dataset = generate_dataset()
