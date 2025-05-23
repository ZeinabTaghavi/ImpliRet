import re
import ast
def filter_generated_feature_parsing_step_A_Uni(generated_response, shopping_list):
    pattern_2 = r'\[\s*\{\s*"item"\s*:\s*"[^"]+"\s*,\s*"brand"\s*:\s*"[^"]+"\s*,\s*"price"\s*:\s*\d+\s*\}(?:\s*,\s*\{\s*"item"\s*:\s*"[^"]+"\s*,\s*"brand"\s*:\s*"[^"]+"\s*,\s*"price"\s*:\s*\d+\s*\})*\s*\]'

    extracted_feature_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        matches = re.findall(pattern_2, str(text).replace("\\n", '').replace("\\", '').replace("\n", ''))
        shopping = shopping_list[idx]
        if len(matches) > 0:
            extracted_item = [(match) for match in matches]
            correct_price_1 = shopping['final_price']
            extracted_prices = str(extracted_item)
            
            if str(correct_price_1) in extracted_prices:
                extracted_feature_list[idx] = extracted_item
                mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
            else:
                mistaken_extracted_idx.append(idx)
                extracted_feature_list[idx] = '-'
        else:
            mistaken_extracted_idx.append(idx)
            extracted_feature_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_feature_list


def filter_generated_feature_parsing_step_A_Multi(generated_response, shopping_list):
    """Filter and validate extracted features from step 2."""
    pattern_2 = r'\[\s*\{\s*"item"\s*:\s*"[^"]+"\s*,\s*"brand"\s*:\s*"[^"]+"\s*,\s*"model"\s*:\s*\d+\s*,\s*"price"\s*:\s*\d+\s*\}(?:\s*,\s*\{\s*"item"\s*:\s*"[^"]+"\s*,\s*"brand"\s*:\s*"[^"]+"\s*,\s*"model"\s*:\s*\d+\s*,\s*"price"\s*:\s*\d+\s*\})*\s*\]'
    extracted_feature_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        matches = re.findall(pattern_2, str(text).replace("\\n", '').replace("\\", ''))
        shopping = shopping_list[idx]
        
        if len(matches) > 0:
            try:

                extracted_item = [(match) for match in matches]
                try:

                    if type(extracted_item) == list:
                        extracted_dictionary = ast.literal_eval(extracted_item[0].replace("\\", '').replace("\n", '').replace("\\n", ''))
                    else:
                        extracted_dictionary = ast.literal_eval(extracted_item.replace("\\", '').replace("\n", '').replace("\\n", ''))

                    correct_price = shopping['question']

                    if len(extracted_dictionary) == 2:
                        if str(extracted_dictionary[0]['price']) in correct_price or str(extracted_dictionary[1]['price']) in correct_price:
                            extracted_feature_list[idx] = extracted_dictionary
                            mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
                        else:
                            print('--------------------------------')
                            print(extracted_dictionary)
                            print(correct_price)
                            print(shopping['forum_post'])
                            mistaken_extracted_idx.append(idx)
                            extracted_feature_list[idx] = '-'
                    if len(extracted_dictionary) == 1:
                        if str(extracted_dictionary[0]['price']) in correct_price:
                            extracted_feature_list[idx] = extracted_dictionary
                            mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
                        else:
                            print('--------------------------------')
                            print(extracted_dictionary)
                            print(correct_price)
                            mistaken_extracted_idx.append(idx)
                            extracted_feature_list[idx] = '-'
                except:
                    print('--------------------------------')
                    print('mistaken in generation')
                    print(shopping['question'])
                    print(extracted_item)
                    mistaken_extracted_idx.append(idx)
                    extracted_feature_list[idx] = '-'
            except:
                mistaken_extracted_idx.append(idx)
                extracted_feature_list[idx] = '-'
        else:
            mistaken_extracted_idx.append(idx)
            extracted_feature_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_feature_list


def filter_generated_feature_parsing_step_S_Uni(generated_response, trip_list):
    """Filter and validate generated features from step 2."""
    pattern_2 = r'\[\s*\{\s*"destination"\s*:\s*"[^"]+"\s*,\s*"friends"\s*:\s*"[^"]+"\s*\}(?:\s*,\s*\{\s*"destination"\s*:\s*"[^"]+"\s*,\s*"friends"\s*:\s*"[^"]+"\s*\})*\s*\]'

    extracted_feature_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        matches = re.findall(pattern_2, str(text).replace("\\n", '').replace("\\", '').replace("\n", ''))
        trip = trip_list[idx]
        
        if len(matches) > 0:
            extracted_item = [(match) for match in matches]
            correct_destination = trip['trip_destination'].replace("'","")
            correct_friends = trip['trip_friends']
            extracted_destinations = str(extracted_item).replace("'","").replace('/','').replace('\\',"")
            if sum([section in extracted_destinations for section in str(correct_destination).split(' ')]) > 0 \
                and correct_friends in extracted_destinations:
                extracted_feature_list[idx] = extracted_item
                mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
            else:
                print('------------ Extracted Destinations --------------------')
                print(sum([section in extracted_destinations for section in str(correct_destination).split(' ')]) > 0)
                print(correct_friends in extracted_destinations)
                print(extracted_destinations)
                print(correct_destination)
                print(correct_friends)
                mistaken_extracted_idx.append(idx)
                extracted_feature_list[idx] = '-'
        else:
            print('not matched')
            print(text)
            mistaken_extracted_idx.append(idx)
            extracted_feature_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_feature_list

def filter_generated_feature_parsing_step_S_Multi(generated_response, trip_list):
    extracted_feature_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        if sum([section in text.lower() for section in str(trip_list[idx]['forum_post'][2].lower()).split(' ')]) > 0 or "Rankås".lower() in text.lower():
            extracted_feature_list[idx] = trip_list[idx]['forum_post'][2]
            mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
        else:
            print('--------------------------------')
            print(text)
            print(trip_list[idx]['forum_post'][2])
            print(sum([section in text.lower() for section in str(trip_list[idx]['forum_post'][2]).split(' ')]))
            print('--------------------------------')
            mistaken_extracted_idx.append(idx)
            extracted_feature_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_feature_list


def filter_generated_works_parsing_step_T_Multi(generated_response, forum_posts):
    """Filter and validate extracted works from step 2."""
    pattern_2 = r'\{["\'\s]*work["\'\s]*:[^}]+,["\'\s]*days["\'\s]*:[^}][^}]+\}'
    extracted_work_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        matches = re.findall(pattern_2, str(text).replace("\\n", '').replace("\\", ''))
        post = forum_posts[idx]
        
        if len(matches) > 0:
           
            extracted_item = [(match) for match in matches]
            try:
                if type(extracted_item) == list:
                    work_dictionary = ast.literal_eval(extracted_item[0])
                else:
                    work_dictionary = ast.literal_eval(extracted_item)

                correct_day = post['offset_days'].split(' ')[0].strip()
                extracted_day = work_dictionary['days'].split(' ')[0]

                if correct_day in extracted_day:
                    extracted_work_list[idx] = work_dictionary
                    mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]
                else:
                    mistaken_extracted_idx.append(idx)
                    extracted_work_list[idx] = '-'
            except:
                mistaken_extracted_idx.append(idx)
                extracted_work_list[idx] = '-'
        else:
            mistaken_extracted_idx.append(idx)
            extracted_work_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_work_list


def filter_generated_works_parsing_step_T_Uni(generated_response, schedules):

    pattern_2 = r'\{["\'\s]*work["\'\s]*:[^}]+,["\'\s]*days["\'\s]*:[^}]+,["\'\s]*hours["\'\s]*:[^}]+\}'
    extracted_work_list = generated_response.copy()
    mistaken_extracted_idx = []
    
    for idx, text in enumerate(generated_response):
        matches = re.findall(pattern_2, str(text).replace("\\n", '').replace("\\", ''))
        schedule = schedules[idx]
        
        if len(matches) > 0:
            try:
                extracted_item = [(match) for match in matches]
                try:
                    if type(extracted_item) == list:
                        work_dictionary = ast.literal_eval(extracted_item[0])
                    else:
                        work_dictionary = ast.literal_eval(extracted_item)

                    correct_day = schedule['days']
                    extracted_day = work_dictionary['days']

                    correct_hour = schedule['hours']
                    extracted_hour = work_dictionary['hours']

                    correct_hour_flag = False
                    if correct_hour[0] == extracted_hour[0] and correct_hour[1] == extracted_hour[1]:
                        correct_hour_flag = True

                    correct_day_flag = True
                    for day in set(correct_day):
                        if day not in set(extracted_day):
                            correct_day_flag = False

                    if len(set(correct_day)) != len(set(extracted_day)):
                        correct_day_flag = False

                    if not (correct_day_flag and correct_hour_flag):
                        mistaken_extracted_idx.append(idx)
                    else:
                        extracted_work_list[idx] = work_dictionary
                        mistaken_extracted_idx = [i for i in mistaken_extracted_idx if i != idx]

                except:
                    mistaken_extracted_idx.append(idx)
                    extracted_work_list[idx] = '-'
            except:
                mistaken_extracted_idx.append(idx)
                extracted_work_list[idx] = '-'
        else:
            mistaken_extracted_idx.append(idx)
            extracted_work_list[idx] = '-'
    
    return sorted(set(mistaken_extracted_idx)), extracted_work_list
