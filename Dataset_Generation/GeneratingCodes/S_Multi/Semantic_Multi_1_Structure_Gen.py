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


def generate_user_trip_destination(country_list, num_items):
    country_list_names = sorted(list(country_list.keys()))
    country_list_names = random.sample(country_list_names, num_items)
    user_trip_destinations = []
    for country in country_list_names:
        user_trip_destinations.append(random.choice(country_list[country]))

    return user_trip_destinations


def generate_message_for_each_user(list_of_users, list_of_destinations, year=2024):
    messages = []
    all_countries = []
    for user, destination in zip(list_of_users, list_of_destinations):
        random_date = f"{year}-{random.randint(1,12)}-{random.randint(1,28)}"
        destination_name = destination['itemLabel']
        destination_country = destination['countryLabel']
        destination_type = destination['instanceOf']
        all_countries.append(destination_country)
        message = {
            "forum_post": (random_date, user, destination_name),
            "destination_type": destination_type,
            "question": f"Who has been to {destination_country}?",
            "answer": user
        }
        
        messages.append(message)
    assert (len(list(set(all_countries))) == len(list_of_destinations))
    return messages


def generate_dataset(num_items=20):

    with open("./Dataset_Generation/Dataset_Helping/names.txt", "r") as file:
        names = ' '.join([line.strip() for line in file.readlines()])
        names_list = sorted(list(ast.literal_eval(names)))

    with open("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_8_gpt_unique_locations.jsonl", "r") as f:
        wikidata_samples = [json.loads(line) for line in f]

    with open("./Dataset_Generation/Dataset_Helping/S_Multi/S_Multi_forum.jsonl", "r") as f:
        forum_topics = [json.loads(line) for line in f]


    dataset = []
    shuffled_names_list = names_list.copy()
    num_items = 20

    country_list = {}
    for sample in wikidata_samples:
        if sample['countryLabel'] not in country_list.keys():
            country_list[sample['countryLabel']] = [sample]
        else:
            country_list[sample['countryLabel']].append(sample)

    # We'll generate enough unique users for each topic; here, 20 per topic.
    for topic in forum_topics:
        dataset_row = {
            'topic': topic["topic"],
            'forum_question': topic["forum_question"],
        }

        list_of_users = random.sample(shuffled_names_list, num_items)
        list_of_destinations = generate_user_trip_destination(country_list, num_items)
        messages = generate_message_for_each_user(list_of_users, list_of_destinations)
        dataset_row["posts"] = messages
        dataset.append(dataset_row)

    # Save dataset
    output_dir = Path("./Dataset_Generation/Dataset_Helping/S_Multi")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"S_Multi_Structured.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Step1: Semantic Multi Dataset Structure saved to {output_file}") 


if __name__ == "__main__":
    generate_dataset()