import json
import argparse
import ast
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def save_jsonl(data, filename):
    """Save data to a JSONL file."""
    with open(filename, "w+", encoding="utf-8") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")

def parse_date(date_str):
    """Parse a 'YYYY-MM-DD' string into a date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def format_date(dt):
    """Format a date object back into a 'YYYY-MM-DD' string."""
    return dt.strftime("%Y-%m-%d")

def random_date_between(start_date, end_date):
    """Return a random date between two date objects."""
    delta = (end_date - start_date).days
    rand_days = random.randint(0, delta)
    return start_date + timedelta(days=rand_days)

def generate_work_dates(n, overall_start, overall_end):
    """
    Generate a chain of n work dates with constraints.
    
    Args:
        n: Number of dates to generate
        overall_start: Start date boundary
        overall_end: End date boundary
        
    Returns:
        List of dates where each subsequent date is 2-7 days after previous,
        and final date is at least 14 days before overall_end
    """
    last_allowed = overall_end - timedelta(days=14)
    min_total_gap = (n - 1) * 2
    max_initial = last_allowed - timedelta(days=min_total_gap)
    start = random_date_between(overall_start, max_initial)
    dates = [start]
    
    for i in range(1, n):
        gap = random.randint(2, 7)
        next_date = dates[-1] + timedelta(days=gap)
        dates.append(next_date)
    return dates

def generate_message_dates(work_dates):
    """
    Generate message dates corresponding to work dates.
    
    Args:
        work_dates: List of work date objects
        
    Returns:
        List of message dates where each date corresponds to next work date,
        except last message which is within 1 week after last work date
    """
    message_dates = []
    for i in range(len(work_dates) - 1):
        message_dates.append(work_dates[i+1])
    
    last_work = work_dates[-1]
    message_dates.append(random_date_between(last_work, last_work + timedelta(days=7)))
    return message_dates

# ------------------------------------------------------------
# Main Generation Function
# ------------------------------------------------------------

def generate_dataset(num_users = 50, num_items = 30, max_price = 3000):
    """Generate the temporal multi-task dataset structure."""
    
    # Load names data
    names_list = []
    with open("./Dataset_Generation/Dataset_Helping/names.jsonl", "r") as file:
        for line in file:
            names_list.append(json.loads(line.strip()))

    # Load forum topics
    forum_topics = []
    with open("./Dataset_Generation/Dataset_Helping/T_Multi/T_Multi_forum.jsonl", "r") as file:
        for line in file:
            forum_topic = json.loads(line.strip())
            forum_topics.append(forum_topic)

    # Set date boundaries
    OVERALL_START = parse_date("2024-01-01")
    OVERALL_END = parse_date("2024-12-31")

    # ------------------------------------------------------------
    # Generate dataset
    # ------------------------------------------------------------

    dataset = []
    shuffled_names_list = names_list.copy()

    for topic in forum_topics:
        dataset_row = {
            'topic': topic["topic"],
            'forum_question': topic["forum_question"],
            "posts": []
        }
        
        work_dates = generate_work_dates(30, OVERALL_START, OVERALL_END)
        topic_users = random.sample(shuffled_names_list, 30)

        message_dates = generate_message_dates(work_dates)
        
        for i in range(30):
            work_date = datetime.strptime(format_date(work_dates[i]), "%Y-%m-%d").strftime("%B %d, %Y")
            message_date = format_date(message_dates[i])
            offset = (message_dates[i] - work_dates[i]).days
            user = topic_users[i]
            forum_item = topic["items"][i]
            forum_post = f"{forum_item}"
            question_template = topic["question"]
            question = question_template.replace("{date}", work_date)
            answer = user['name']
            record = {
                "forum_post": (message_date, user, forum_post),
                "offset_days": f"{offset} days ago",
                "question": question,
                "answer": answer
            }
            dataset_row["posts"].append(record)

        dataset.append(dataset_row)
        
    # ------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------

    output_dir = Path("./Dataset_Generation/Dataset_Helping/T_Multi")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"T_Multi_Structured.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Step1: Temporal Multi Dataset Structure saved to {output_file}")
        
        
if __name__ == "__main__":
    generate_dataset()