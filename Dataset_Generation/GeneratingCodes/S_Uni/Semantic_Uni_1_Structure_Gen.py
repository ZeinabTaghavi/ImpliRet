import random
from datetime import timedelta, date, datetime
import ast
from pathlib import Path
import json

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)


def generate_user_trip_destination(country_list, num_items):
    """
    Generate random trip destinations for a user from a list of countries.
    
    Args:
        country_list: Dictionary mapping countries to lists of destination details
        num_items: Number of destinations to generate
        
    Returns:
        List of randomly selected trip destinations
    """
    # Get sorted list of country names and randomly sample
    country_list_names = sorted(list(country_list.keys()))
    country_list_names = random.sample(country_list_names, num_items)
    
    # For each country, randomly select one destination
    user_trip_destinations = []
    for country in country_list_names:
        user_trip_destinations.append(random.choice(country_list[country]))

    return user_trip_destinations

def generate_dataset(users_num=50, num_items=30):
    """
    Generate complete semantic dataset of user trips.
    
    Args:
        users_num: Number of users to generate data for
        num_items: Number of trips per user
    """
    # ------------------------------------------------------------
    # Load Names and Trip Data
    # ------------------------------------------------------------
    
    # Load list of user names
    names_list = []
    with open("./Dataset_Generation/Dataset_Helping/names.jsonl", "r") as file:
        for line in file:
            names_list.append(json.loads(line.strip()))

    # Load destination data from Wikidata
    with open("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_8_gpt_unique_locations.jsonl", "r") as f:
        wikidata_samples = [json.loads(line) for line in f]

    # Load the travel purposes
    with open("./Dataset_Generation/Dataset_Helping/S_Uni/S_Uni_Base_JSON.json", "r") as f:
        travel_purposes = json.load(f)['travel_purposes']


    # ------------------------------------------------------------
    # Organize Destinations by Country
    # ------------------------------------------------------------
    
    # Create dictionary mapping countries to their destinations
    country_list = {}
    for sample in wikidata_samples:
        if sample['countryLabel'] not in country_list.keys():
            country_list[sample['countryLabel']] = [sample]
        else:
            country_list[sample['countryLabel']].append(sample)

    # ------------------------------------------------------------
    # Generate User Trip Data
    # ------------------------------------------------------------
    
    # Randomly select users
    users = random.sample(names_list, users_num)
    user_dataset = {}

    # Generate trip data for each user
    for user in users:
        user_dataset[user['name']] = {"user_1": user, "trip_info": []}
        travel_purpose = random.sample(travel_purposes, num_items)
        assert len(set(travel_purpose)) == len(travel_purpose), "Travel purposes are not unique"

        # Generate destinations for this user
        user_trips_items = generate_user_trip_destination(country_list, num_items)

        # Select friends and companions for trips
        friends = random.sample([f for f in names_list if f != user], int(num_items * 2))
        users_2 = random.sample(friends, num_items)  # Primary companions
        
        # Create trip entries
        for i in range(num_items):
            user_2 = users_2[i]
            trip = user_trips_items[i]
            purpose = travel_purpose[i]

            trip_info = {
                "user_2": user_2,
                "trip_destination": trip['itemLabel'],
                "trip_purpose": purpose,
                "trip_country": trip['countryLabel'],
                "type_of_location": trip['instanceOf']
            }
            user_dataset[user['name']]["trip_info"].append(trip_info)

    # ------------------------------------------------------------
    # Save Dataset
    # ------------------------------------------------------------
    
    output_file = './Dataset_Generation/Dataset_Helping/S_Uni/S_Uni_Structured.jsonl'
    with open(output_file, 'w') as f:
        for user, data in user_dataset.items():
            line = {'user_1': data['user_1'], 'trip_info': data['trip_info']}
            f.write(json.dumps(line) + '\n')
    
    print(f"Step1: Semantic Uni Dataset Structure saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()