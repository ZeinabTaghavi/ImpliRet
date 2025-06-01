import random
from datetime import timedelta, date, datetime
import ast
from pathlib import Path
import json

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)

# ------------------------------------------------------------
# Helper Functions for Schedule Generation
# ------------------------------------------------------------

def initial_free_intervals(day_start, day_end):
    """Return the initial free interval for a day: [(DAY_START, DAY_END)]."""
    return [(day_start, day_end)]

def schedule_in_interval(free_interval, duration, max_block):
    """
    Schedule an activity within a free interval.
    
    Args:
        free_interval: Tuple of (start, end) times
        duration: Desired activity duration
        max_block: Maximum block size allowed
        
    Returns:
        Tuple of (scheduled_start, scheduled_end) or None if not possible
    """
    start, end = free_interval
    interval_length = end - start
    
    # Define candidate blocks from the start and end
    block_start = start
    block_length_start = min(max_block, interval_length)
    block_end = end  
    block_length_end = min(max_block, interval_length)
    
    choice = random.choice(["start", "end"])
    if choice == "start":
        available_delay = block_length_start - duration
        if available_delay < 0:
            return None
        delay = random.randint(0, available_delay)
        scheduled_start = block_start + delay
        scheduled_end = scheduled_start + duration
    else:
        available_delay = block_length_end - duration
        if available_delay < 0:
            return None
        delay = random.randint(0, available_delay)
        scheduled_end = block_end - delay
        scheduled_start = scheduled_end - duration
    return (scheduled_start, scheduled_end)

def update_interval(free_interval, scheduled_slot):
    """
    Update free intervals after scheduling an activity.
    
    Args:
        free_interval: Original free interval (start, end)
        scheduled_slot: Scheduled activity slot (start, end)
        
    Returns:
        List of remaining free intervals
    """
    s_free, e_free = free_interval
    s_slot, e_slot = scheduled_slot
    new_intervals = []
    if s_slot > s_free:
        new_intervals.append((s_free, s_slot))
    if e_slot < e_free:
        new_intervals.append((e_slot, e_free))
    return new_intervals

def intersect_intervals(intervals):
    """
    Find intersection of multiple intervals.
    
    Args:
        intervals: List of (start, end) intervals
        
    Returns:
        Tuple of (start, end) for intersection, or None if no overlap
    """
    common_start = max(interval[0] for interval in intervals)
    common_end = min(interval[1] for interval in intervals)
    if common_end > common_start:
        return (common_start, common_end)
    else:
        return None

# ------------------------------------------------------------
# Main Schedule Generation Function
# ------------------------------------------------------------

def making_schedule(user, users, WINDOW_DAYS, W_repeat, W_once, min_duration, max_duration, max_block, day_start, day_end):
    """
    Generate a schedule for a single user.
    
    Args:
        user: User to generate schedule for
        users: List of all users
        WINDOW_DAYS: Number of days to schedule
        W_repeat: List of repeating activities
        W_once: List of one-time activities
        min_duration: Minimum activity duration
        max_duration: Maximum activity duration
        max_block: Maximum block size
        day_start: Start hour of day
        day_end: End hour of day
        
    Returns:
        List of scheduled activities
    """
    user_scheduled_correctly = False
    scheduling_attempts = 0
    
    while not user_scheduled_correctly:
        scheduling_attempts += 1
        # Initialize free intervals for each day
        free_times = {day: initial_free_intervals(day_start, day_end) for day in range(1, WINDOW_DAYS+1)}
        schedule = []
        
        # -----------------------------
        # 1. Schedule Repeating-Sequential Activities
        # -----------------------------
        RS_works = random.sample(W_repeat, 3)
        RS_durations = [3, 3, 4]  # consecutive days
        for work, duration in zip(RS_works, RS_durations):
            valid_start_days = list(range(1, WINDOW_DAYS - duration + 2))
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                start_day = random.choice(valid_start_days)
                day_sequence = list(range(start_day, start_day + duration))
                candidate_intervals = []
                for day in day_sequence:
                    if free_times[day]:
                        candidate_intervals.append(free_times[day][0])
                    else:
                        candidate_intervals = []
                        break
                if not candidate_intervals:
                    attempts += 1
                    continue
                common_interval = intersect_intervals(candidate_intervals)
                if common_interval is None or (common_interval[1] - common_interval[0]) < min_duration:
                    attempts += 1
                    continue
                dur = random.randint(min_duration, max_duration)
                slot = schedule_in_interval(common_interval, dur, max_block)
                if slot is None:
                    attempts += 1
                    continue
                # Update free intervals for each day in the sequence
                for day in day_sequence:
                    new_intervals = []
                    for interval in free_times[day]:
                        if interval[0] <= common_interval[0] and interval[1] >= common_interval[1]:
                            new_intervals.extend(update_interval(interval, slot))
                        else:
                            new_intervals.append(interval)
                    free_times[day] = new_intervals
                if duration == 3:
                    qs_day = day_sequence[1]
                else:
                    qs_day = random.choice(day_sequence[1:3])
                s_slot, e_slot = slot
                if e_slot - s_slot > 1:
                    qs_hour = random.randint(s_slot + 1, e_slot - 1)
                else:
                    qs_hour = s_slot
                user_2 = random.choice([s_u for s_u in users if s_u['name'] != user['name']])
                schedule.append({
                    "user_2": user_2,
                    "work": work,
                    "activity_type": "Repeating-Sequential",
                    "days": day_sequence,
                    "hours": slot,
                    "question_time": (qs_day, qs_hour)
                })
                placed = True
            if not placed:
                print(f"Warning: Could not place repeating-sequential work {work} for {user}")
        
        # -----------------------------
        # 2. Schedule Repeating-Non-Sequential Activities
        # -----------------------------
        remaining_repeat = [w for w in W_repeat if w not in RS_works]
        RN_works = random.sample(remaining_repeat, 3)
        RN_durations = [2, 3, 2]  # non-consecutive days
        for work, duration in zip(RN_works, RN_durations):
            placed = False
            attempts = 0
            valid_days = list(range(1, WINDOW_DAYS+1))
            while not placed and attempts < 100:
                day_sequence = sorted(random.sample(valid_days, duration))
                candidate_intervals = []
                for day in day_sequence:
                    if free_times[day]:
                        candidate_intervals.append(free_times[day][0])
                    else:
                        candidate_intervals = []
                        break
                if not candidate_intervals:
                    attempts += 1
                    continue
                common_interval = intersect_intervals(candidate_intervals)
                if common_interval is None or (common_interval[1] - common_interval[0]) < min_duration:
                    attempts += 1
                    continue
                dur = random.randint(min_duration, max_duration)
                slot = schedule_in_interval(common_interval, dur, max_block)
                if slot is None:
                    attempts += 1
                    continue
                # Update free intervals for each day in the sequence
                for day in day_sequence:
                    new_intervals = []
                    for iv in free_times[day]:
                        if iv[0] <= common_interval[0] and iv[1] >= common_interval[1]:
                            new_intervals.extend(update_interval(iv, slot))
                        else:
                            new_intervals.append(iv)
                    free_times[day] = new_intervals
                if len(day_sequence) >= 2:
                    qs_day = day_sequence[1]
                else:
                    qs_day = day_sequence[0]
                s_slot, e_slot = slot
                if e_slot - s_slot > 1:
                    qs_hour = random.randint(s_slot + 1, e_slot - 1)
                else:
                    qs_hour = s_slot
                user_2 = random.choice([s_u for s_u in users if s_u['name'] != user['name']])
                schedule.append({
                    "user_2": user_2,
                    "work": work,
                    "activity_type": "Repeating-Non-Sequential",
                    "days": day_sequence,
                    "hours": slot,
                    "question_time": (qs_day, qs_hour)
                })
                placed = True
                break
            if not placed:
                print(f"Warning: Could not place repeating-non-sequential work {work} for {user}")
        
        # -----------------------------
        # 3. Schedule One-Time Activities
        # -----------------------------
        OT_works = random.sample(W_once, 9)
        for work in OT_works:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                day = random.randint(1, WINDOW_DAYS)
                if not free_times[day]:
                    attempts += 1
                    continue
                interval = free_times[day][0]
                dur = random.randint(min_duration, max_duration)
                slot = schedule_in_interval(interval, dur, max_block)
                if slot is None:
                    attempts += 1
                    continue
                new_ints = []
                for iv in free_times[day]:
                    if iv == interval:
                        new_ints.extend(update_interval(iv, slot))
                    else:
                        new_ints.append(iv)
                free_times[day] = new_ints
                s_slot, e_slot = slot
                if e_slot - s_slot > 1:
                    qs_hour = random.randint(s_slot + 1, e_slot - 1)
                else:
                    qs_hour = s_slot
                user_2 = random.choice([s_u for s_u in users if s_u['name'] != user['name']])
                schedule.append({
                    "user_2": user_2,
                    "work": work,
                    "activity_type": "One-Time",
                    "days": [day],
                    "hours": slot,
                    "question_time": (day, qs_hour)
                })
                placed = True
                break
            if not placed:
                print(f"Warning: Could not place one-time work {work} for {user}")
        
        if len(schedule) == 15:
            user_scheduled_correctly = True

        if scheduling_attempts == 100:
            print(f"Warning: Could not schedule activities for {user}")
            raise Exception(f"Could not schedule activities for {user}")

    return schedule

# ------------------------------------------------------------
# Dataset Generation Function
# ------------------------------------------------------------

def generate_dataset(num_users = 50, num_items = 30, window_days = 14, day_start = 7, day_end = 19, min_duration = 2, max_duration = 4, max_block = 4):
    """
    Generate complete temporal dataset.
    
    Args:
        num_users: Number of users to generate schedules for
        num_items: Number of activities per user
        window_days: Number of days in schedule window
        day_start: Start hour of day
        day_end: End hour of day
        min_duration: Minimum activity duration
        max_duration: Maximum activity duration
        max_block: Maximum block size
    """
    # Define overall period boundaries
    START_YEAR = 2020
    END_YEAR = 2024

    # ------------------------------------------------------------
    # Load Activity Lists and Names
    # ------------------------------------------------------------
    with open("./Dataset_Generation/Dataset_Helping/T_Uni/works_Temp_Uni_Once.txt", "r") as file:
        phrases = ' '.join([line.strip() for line in file.readlines()])
        W_once_list = sorted(list(ast.literal_eval(phrases)))

    with open("./Dataset_Generation/Dataset_Helping/T_Uni/works_Temp_Uni_Repeat.txt", "r") as file:
        phrases = ' '.join([line.strip() for line in file.readlines()])
        W_repeat_list = sorted(list(ast.literal_eval(phrases)))

    names_list = []
    with open("./Dataset_Generation/Dataset_Helping/names.jsonl", "r") as file:
        for line in file:
            names_list.append(json.loads(line.strip()))

    # ------------------------------------------------------------
    # Generate Basic Schedules
    # ------------------------------------------------------------
    users = random.sample(names_list, num_users)
    user_schedules = {}

    for user in users:
        W_once_1 = random.sample(W_once_list, 25)
        W_once_2 = [w for w in W_once_list if w not in W_once_1]
        W_repeat_1 = random.sample(W_repeat_list, 10)
        W_repeat_2 = [w for w in W_repeat_list if w not in W_repeat_1]
        
        schedule_1 = making_schedule(user, users, window_days, W_repeat_1, W_once_1, min_duration, max_duration, max_block, day_start, day_end)
        schedule_2 = making_schedule(user, users, window_days, W_repeat_2, W_once_2, min_duration, max_duration, max_block, day_start, day_end) 

        # Adjust days for second schedule
        for i in range(len(schedule_2)):
            schedule_2[i]['days'] = [day + 14 for day in schedule_2[i]['days']]
            schedule_2[i]['question_time'] = (schedule_2[i]['question_time'][0]+14 , schedule_2[i]['question_time'][1])

        schedule = schedule_1 + schedule_2
        user_schedules[user['name']] = {
            "user_1": user,
            "schedule": schedule
        }

    # ------------------------------------------------------------
    # Add Actual Dates
    # ------------------------------------------------------------
    overall_start = date(START_YEAR, 1, 1)
    overall_end = date(END_YEAR, 12, 31)
    total_days = (overall_end - overall_start).days
    segment_length = total_days // num_users

    user_start_dates = {}
    sorted_users = sorted(user_schedules.keys())

    # Assign start dates for each user
    for i, user in enumerate(sorted_users):
        seg_start = overall_start + timedelta(days=i * segment_length)
        seg_end = seg_start + timedelta(days=segment_length - 28)
        
        if seg_end < seg_start:
            raise ValueError(f"Segment {i} has no valid start date.")
            
        delta = (seg_end - seg_start).days
        offset = random.randint(0, delta) if delta > 0 else 0
        base_date = seg_start + timedelta(days=offset)
       
        user_start_dates[user] = base_date

    # Update schedules with actual dates
    for user, schedule_info in user_schedules.items():
        base_date = user_start_dates[user]
        for activity in schedule_info["schedule"]:
            actual_days = []
            for day in activity["days"]:
                actual_date = base_date + timedelta(days=day)
                actual_days.append(actual_date.strftime("%Y-%m-%d"))
            activity["days"] = actual_days
            
            qs_day_index, qs_hour = activity["question_time"]
            actual_qs_date = base_date + timedelta(days=qs_day_index)
            activity["question_time"] = (actual_qs_date.strftime("%Y-%m-%d"), qs_hour)

    # ------------------------------------------------------------
    # Add Question Times
    # ------------------------------------------------------------
    for user in user_schedules:
        question_times = [activity["question_time"] for activity in user_schedules[user]["schedule"]]
        message_times = question_times.copy()
        
        valid_shuffle = False
        while not valid_shuffle:
            random.shuffle(message_times)
            valid_shuffle = all(q != m for q, m in zip(question_times, message_times))
        
        for activity, message_time in zip(user_schedules[user]["schedule"], message_times):
            activity["message_time"] = message_time

            # Calculate offset days between message time and each activity day
            message_date = datetime.strptime(message_time[0], "%Y-%m-%d")
            offset_days = []

            if activity["activity_type"] != "Repeating-Sequential":
                for activity_day in activity["days"]:
                    activity_date = datetime.strptime(activity_day, "%Y-%m-%d")
                    # Calculate the difference in days
                    delta = (activity_date - message_date).days
                    if delta < -1:
                        offset_days.append(f"{-1 * delta} days ago")
                    elif delta == -1:
                        offset_days.append(f"Yesterday")
                    elif delta == 0:
                        offset_days.append(f"Today")
                    elif delta == 1:
                        offset_days.append(f"Tomorrow")
                    else:
                        offset_days.append(f"{delta} days later")
            else:
                min_date = min(activity["days"])
                delta_min = (datetime.strptime(min_date, "%Y-%m-%d") - message_date).days
                if delta_min < -1:
                    offset_days.append(f"Started {-1 * delta_min} days ago, for {len(activity['days'])} consecutive days")
                elif delta_min == -1:
                    offset_days.append(f"Started yesterday, for {len(activity['days'])} consecutive days")
                elif delta_min == 0:
                    offset_days.append(f"Starting today, for {len(activity['days'])} consecutive days")
                elif delta_min == 1:
                    offset_days.append(f"Starting tomorrow, for {len(activity['days'])} consecutive days")
                elif delta_min == 1:
                    offset_days.append(f"Starting tomorrow, for {len(activity['days'])} consecutive days")
                else:
                    offset_days.append(f"Starting in {delta_min} days, for {len(activity['days'])} consecutive days")

            activity["offset_days"] = offset_days

    # ------------------------------------------------------------
    # Save Dataset
    # ------------------------------------------------------------
    output_dir = Path("./Dataset_Generation/Dataset_Helping/T_Uni")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"T_Uni_Structured.jsonl"

    with open(output_file, 'w') as f:
        for user, schedule_data in user_schedules.items():
            schedule_with_user = schedule_data.copy()
            schedule_with_user['user_1'] = schedule_with_user['user_1']
            for schedule in schedule_with_user['schedule']:
                schedule["message_time"] = (
                    schedule["message_time"][0], 
                    datetime.strptime(schedule["message_time"][0], "%Y-%m-%d").strftime("%A").lower(),
                    schedule["message_time"][1]
                )
            json.dump(schedule_with_user, f)
            f.write('\n')

    print(f"Step1: Temporal Uni Dataset Structure saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()