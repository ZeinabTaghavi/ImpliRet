import random
from datetime import timedelta, date, datetime
import ast
from pathlib import Path
import json

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)

# def generate_distinct_dates(n, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 12, 31)):
#     """
#     Generate n distinct random dates between start_date and end_date.
    
#     Args:
#         n: Number of distinct dates to generate
#         start_date: Start date (default: 2024-01-01)
#         end_date: End date (default: 2024-12-31)
        
#     Returns:
#         List of n distinct dates in ISO format in ascending order
#     """
#     # Calculate total number of possible days
#     total_days = (end_date - start_date).days + 1
    
#     # Ensure n doesn't exceed possible days
#     if n > total_days:
#         raise ValueError(f"Cannot generate {n} distinct dates in a {total_days} day period")
        
#     # Generate n random distinct days
#     random_days = sorted(random.sample(range(total_days), n))
    
#     # Convert to actual dates in ISO format
#     dates = [(start_date + timedelta(days=day)).strftime("%Y-%m-%d") for day in random_days]

#     return dates

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
    with open("./Dataset_Generation/Dataset_Helping/names.txt", "r") as file:
        names = ' '.join([line.strip() for line in file.readlines()])
        names_list = sorted(list(ast.literal_eval(names)))

    # Load destination data from Wikidata
    with open("./Dataset_Generation/Dataset_Helping/S_Both/Wikidata_scored_Records/wikidata_step_8_gpt_unique_locations.jsonl", "r") as f:
        wikidata_samples = [json.loads(line) for line in f]

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
        user_dataset[user] = {"user_1": user, "trip_info": []}

        # Generate destinations for this user
        user_trips_items = generate_user_trip_destination(country_list, num_items)

        # Select friends and companions for trips
        friends = random.sample([f for f in names_list if f != user], int(num_items * 2))
        users_2 = random.sample(friends, num_items)  # Primary companions
        trip_friends = [f for f in friends if f not in users_2]  # Additional friends
        
        # Create trip entries
        for i in range(num_items):
            user_2 = users_2[i]
            trip = user_trips_items[i]
            trip_friend = trip_friends[i]

            trip_info = {
                "user_2": user_2,
                "trip_destination": trip['itemLabel'],
                "trip_friends": trip_friend,
                "trip_country": trip['countryLabel'],
                "type_of_location": trip['instanceOf']
            }
            user_dataset[user]["trip_info"].append(trip_info)

    # ------------------------------------------------------------
    # Save Dataset
    # ------------------------------------------------------------
    
    output_file = './Dataset_Generation/Dataset_Helping/S_Uni/S_Uni_Structured.jsonl'
    with open(output_file, 'w') as f:
        for user, data in user_dataset.items():
            line = {'user_1': user, 'trip_info': data['trip_info']}
            f.write(json.dumps(line) + '\n')
    
    print(f"Step1: Semantic Uni Dataset Structure saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()