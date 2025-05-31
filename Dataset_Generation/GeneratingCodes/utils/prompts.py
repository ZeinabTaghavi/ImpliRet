ALL_PROMPTS = {

# A_Multi
"PROMPTS_A_Multi": {
'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural response to a forum question.

**Input**
- `topic`: A short topic of the forum discussion.
- `forum_question`: A base question posted in the forum.
- `user`: a dictionary with the following keys:
  - `name`: Name of the user.
  - `persona`: Persona of the user.
- `forum_post`: A list of three sentences:
  1. The price of an item from a certain brand's model to dollars (e.g., "Gaming Chairs from Secretlab model 2019: 1650 dollars").
  2. The price of another item from the same brand but a different model, the price is described in relative form to the first item (e.g., "Secretlab, model 2019: 2.5 times more expensive than model 2016").
  3. A sentence stating which model of the brand was ultimately purchased (e.g., "model 2016 was purchased").


**Requirements**
- In the response, you should answer the "forum_question" by using the information in "forum_post".
- You can use the information in user["persona"] that is about the user['name'] to make the response more natural.
- Preserve the numeric references (prices, multipliers, etc.).
- You can write the numbers as words, but do not change the value at all (e.g., 3.5 can be mentioned as "three and a half").
- Write the relative price in a natural way (e.g., "Secretlab, model 2019: 2.5 times more expensive than model 2016" can be mentioned as "The Secretlab 2019 model costs two and a half times as much as the 2016 model.").
- Explicitly mention the brand and model references, or the model was purchased.
- Only mention the information once in the response.
- Make sure that you generate grammatically correct sentences.
- Optionally include a reason for choosing the second brand.
- Your generated answer must be coherent and make the Answer natural as a human is really answering the `forum_question` in 5 sentences.


**Output Format**
Only 1 line of response, without any prefix or suffix.

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify purchased items, their brands, and prices in a conversation transcript.

**Input**
You are given a forum response transcript with 4 sentences as INPUT.


**Requirements**
- Your task is to identify items purchased in the conversation, the brand of those items, and their prices. Note that the cost might be stated explicitly or described in relative terms. In such relative cases, you must calculate the final price numerically based on the information available.
- Identify any items that someone actually buys or mentions buying.
- Determine the brand associated with each purchased item (if specified).
- Extract or compute the price in dollars, performing calculations for relative pricing.
- If no purchases are found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with keys:
- `item` (string): Name of the purchased item.
- `brand` (string): Brand name.
- `model` (integer): Model of the item.
- `price` (integer): Price in dollars.


INPUT: {context}
'''
},

# A_Uni
"PROMPTS_A_Uni": { 

'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural conversation between two people ("user_1" and "user_2") based on a shopping list.

**Input**
- `user_1`: Name of the first user.
- `user_2`: Name of the second user.
- `shopping_type`: Type of shopping.
- `item_to_buy`: The purchased item.
- `bought`: A list of three sentences:
  1. The price of the item in another brand.
  2. The price of the item in the brand bought, relative to the first.
  3. The brand bought.
- `first_sentence`: The first sentence of the conversation.

**Requirements**
- In the conversation, "user_1" will share a shopping information message and should mention that wit as in the 'shopping_type' category and bought the 'item_to_buy', while "user_2" must engage naturally but must not reveal or comment on any shopping or locational information.
- You can use the information in "first_sentence" (modify it if needed) to start the conversation.
- Preserve exact numbers and relative phrasing in the `bought` sentences.
- Explicitly state that "user_1" did not buy from the first brand.
- Explicitly state that "user_1" bought from the second brand.
- "user_2" replies naturally without referencing shopping or numerical details.
- Make sure that you generate grammatically correct sentences.
- Mention exact brands and shopping types as given once in the conversation.
- Optionally include a reason for choosing the second brand.
- The conversation must consist of exactly 10 utterances.
- Each utterance is on its own line.

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the user name (one of the values of `user_1` or `user_2`) and the utterance with a colon

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify purchased items, their brands, and prices in a conversation transcript.

**Input**
Input is a conversation transcript as a list of lines, each:
<user name>: <utterance>

**Requirements**
- Detect items someone buys or mentions buying.
- Determine the brand for each purchased item.
- Extract or compute the price in dollars, performing calculations for relative pricing.
- If no purchases are found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with keys:
- `item` (string): Name of the purchased item.
- `brand` (string): Brand name.
- `price` (integer): Price in dollars.


INPUT: {context}
'''
},

# S_Multi
"PROMPTS_S_Multi": { 
'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural response to a forum question.

**Input**
- `topic`: A short topic of the forum discussion.
- `forum_question`: A base question posted in the forum.
- `forum_post`: The location that the person wants to use for responding to the "forum_question".
- `type_of_location`: The type of location that the person in the post is talking about in "forum_post".
- `first_sentence`: The first sentence of the response.
**Requirements**
- In the response, you should answer the "forum_question" by using the information in "forum_post".
- You can use the information in "first_sentence" (modify it if needed) to start the conversation.
- You should mention that the user was in the location mentioned in "forum_post" and you can use the information in "type_of_location" to know the type of location mentioned in "forum_post".
- Make sure that you generate grammatically correct sentences.
- Only mention the `forum_post` information once in the response.
- Do not alter the location in `forum_post`, use it exactly as it is without any changes.
- Do not mention any other location than the one in `forum_post`
- Optionally include a reason for choosing the second brand.
- Your generated answer must be coherent and make the Answer natural as a human is really answering the `forum_question` in 4 sentences.


**Output Format**
Only 1 line of response, without any prefix or suffix.

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify the location that the person was in.

**Input**
You are given a forum response transcript with 4 sentences as INPUT.


**Requirements**
- Your task is to identify the location that the person was in.
- Extract the exact name and do not change it.
- If no location is found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with the key:
- `location` (string): Name of the location that the person was in.


INPUT: {context}
'''
},

# S_Uni
"PROMPTS_S_Uni": { 
'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural conversation between two people ("user" and "user_2") based on a trip.

**Input**
- `user_1`: Name of the first user.
- `user_2`: Name of the second user.
- `trip_destination`: Destination of the trip.
- `type_of_location`: Type of location.
- `trip_friends`: Friends of the trip.
- `first_sentence`: The first sentence of the conversation.

**Requirements**
- In the conversation, "user_1" will share a trip information message and should mention that it was in the 'trip_destination' with the 'trip_friends', while "user_2" must engage naturally but must not reveal or comment on any trip or locational information.
- You can use the information in "first_sentence" (modify it if needed) to start the conversation.
- "user_2" replies naturally without referencing locational information.
- Do not mention any other locational information in the conversation. Do not mention the country or city of the trip destination.
- Make sure that you have exactly mentioned the 'trip_destination' in the conversation without any other information or changes.
- Mention the 'trip_destination' exactly as it is once in the conversation.
- Make sure that you generate grammatically correct sentences.
- Do not mention the country or city of the 'trip_destination'.
- "type_of_location" is the type of location that the "user_1" has gone to, which can help you make the conversation more natural.
- The conversation must consist of exactly 10 utterances.
- Each utterance is on its own line.

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the user name (one of the values of `user_1` or `user_2`) and the utterance with a colon

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify places that the user has gone or visited, and the people that the user has gone with.

**Input**
Input is a conversation transcript as a list of lines, each:
<user name>: <utterance>

**Requirements**
- Detect places that the user has gone to or visited.
- Determine the people that the user has gone with.
- If no places or people are found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with keys:
- `destination` (string): Name of the place that the user has gone or visited.
- `friends` (string): Name of the people that the user has gone with.

INPUT: {context}
'''
},

# T_Multi
"PROMPTS_T_Multi": { 
'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural response to a forum question.

**Input**
- `topic`: A short topic of the forum discussion.
- `forum_question`: A base question posted in the forum.
- `forum_post`: The item related to the topic that the person wants to use for responding to the "forum_question".
- `offset_days`: The date that the person in the post is talking about in "forum_post".
- `first_sentence`: The first sentence of the response.

**Requirements**
- In the response, you should answer the "forum_question" by using the item mentioned in "forum_post".
- You can use the information in "first_sentence" (modify it if needed) to start the conversation.
- You should mention that the user had done the work on the date mentioned in "offset_days", based on the 'topic', choose a verb that is suitable for the work.
- The work should be done exactly in one day, so avoid using vague temporal references like "until", "by the ...", "completed", "finished", etc.
- Do not alter the `offset_days`, use it exactly as it is without any changes, only you are allowed to say it with words or numbers (e.g., "2 days ago", "two days ago").
- Do not mention any other item than the one in "forum_post".
- Do not mention any date other than the one in "offset_days".
- Only mention the `forum_post` information once in the response.
- Make sure that you generate grammatically correct sentences.
- Your generated answer must be coherent and make the Answer natural as a human is really answering the `forum_question` in 4 sentences.


**Output Format**
Only 1 line of response, without any prefix or suffix.

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify a work-related task described in the user's mention for the forum response and extract its temporal details. Specifically, you should:
	1.	Determine the work task (e.g., the action or project mentioned).
	2.	Identify any temporal expressions referring to when the work is to be performed. Convert relative time expressions (such as "tomorrow", "next week", etc.) into numerical offset_days (e.g., "1 day ago", "2 days ago", "3 days ago", etc.). Be very careful that the relevant dates are correct.


**Input**
You are given a forum response transcript with 4 sentences as INPUT.


**Requirements**
- Your task is to identify the work task and the `offset_days`.
- Mention the `offset_days` as a number with words (e.g., "1 day ago", "2 days ago", "3 days ago", etc.).
- Extract the exact work task and do not change it.
- If no work task or offset_days is found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with the keys:
- `work` (string): The work task.
- `days` (string): The offset_days.


INPUT: {context}
'''
}, 

# T_Uni
"PROMPTS_T_Uni": { 

'CONVERSATION_GENERATION_PROMPT' : '''

**Task**
Generate a natural conversation between two people ("user_1" and "user_2") based on the given schedule.

**Input**
- `user_1`: Name of the first user.
- `user_2`: Name of the second user.
- `work`: The work task.
- `activity_type`: the repeating type of the activity, it can be "One-Time" (that means the user have to do the "work" in the day dated and mentioned "hour"), "Repeating-Sequential" (that means we have a sequential days that the user have to do the "work" in mentioned "hour"), "Repeating-Non-Sequential" (that means the "work" will be done in different days that maybe are not in following days)
- `days`: the dates on which the work has been done.
- `hours`: The hours that the work is to be performed.
- `offset_days`: The list of offset_days of the work, if it is a list of one item, it means the work was done on that day, if it is a list of more than one items, it means the work was done on multiple days.
- `message_time`: The time that the conversation is being sent: [date of the message, day of the week, hour in 24h format]
- `first_sentence`: The first sentence of the conversation.

**Requirements**
- In the conversation, "user_1" will share a schedule information message and should mention that they did the 'work' on a day or days and 'hours', while "user_2" must engage naturally but must not reveal or comment on their schedules.
- You can use the information in "first_sentence" (modify it if needed) to start the conversation.
- You should mention that the 'user_1' did the 'work' on the specific day or days. Mention the day of working relative to today with the help of the corresponding 'offset_days'. For each item of 'offset_days', if the item is negative, it means the work was done before today (e.g, -5 means "5 days ago"); if it is positive, it means the work was done after today (e.g, 4 means "4 days later").
- Make sure that you mentioned all the days in the 'offset_days' list.
- You can also mention the days relative to each other in the conversation, for example, if the 'offset_days' is [2, 6, 12], you can say it in this way: "2 days later, 4 days after that, and 6 days after the second day".
  For example, if the work day is ['2020-01-12'] and the message day is ['2020-01-13']. Here `offset_days` is [-1] : (2020-01-12) - (2020-01-13) = -1. Hence, you can say "yesterday".
  Or if the message_date is ['2021-02-15'] and work days are ['2021-02-17', '2021-02-18', '2021-02-19']. Here `offset_days` are [2, 3, 4].  Hence, you can say "from two days later, for 3 consecutive days". 
  Or if the message_date is ['2021-03-11'] and work days are ['2021-03-16', '2021-03-18'] and the "hours" is (11,15). Here `offset_days` is [5, 7], and the difference is 7-5=2. Hence, you can say "5 days later, and two days after that, from 11 in the morning for 4 hours on both days" or "5 days later, and two days after that, from 11 a.m. for 4 hours on both days"
  Or if the message_date is ['2022-11-10'] and work days are ['2021-11-16', '2021-11-19', '2021-11-23'] and the "hours" is (10,12). Here `offset_days` is [6, 9, 13], and the difference is 9-6=3, and 13-9=4. Hence, you can say "6 days later, and 3 days after that, and 4 days after the second day from 11 in the morning for 4 hours on both days" or "5 days later, and to days after that, from 11 a.m. for 4 hours on both days"
  
- Do not mention any explicit date for working, only mention the days relative to today.
- All the work are being done in the same hour mentioned in 'hours', you should not directly mention the end hour, but make sure that you accurately mention end hour relative to the start hour (e.g., "from 1 p.m. until 3 hours after that" or "from 9 in the morning for three hours").
- Mention the `work` in the conversation exactly as it is (only change the tense if needed).
- Do not change the date information. Ensure that the "hours" you use in the conversation for "work" are correct and accurate. For example, if the work is "updating a work log" and the "message_time" is ("2023-07-21", "Friday", 14), and the "hours" are (7, 10), You can use it like this: "2023-07-21", "Alaina", "I have to update a work log tomorrow from 7 in the morning for three hours."
- The message time is the time at which the conversation is being sent; you can only choose a random minute for the conversation. The format should be like this: "YYYY-MM-DD HH:MM" (e.g., "2024-01-01 12:00").
- "user_2" replies naturally without referencing schedule or numerical details.
- Mention the working schedule once in the conversation.
- Make sure that you generate grammatically correct sentences.
- The conversation must consist of exactly 10 utterances.
- Each utterance is on its own line.

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the message time and the user name (one of the values of `user_1` or `user_2`) with a comma, and separate the user name and the utterance with a colon.

INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify a work-related task described in the conversation and extract its temporal details.

**Input**
Input is a conversation transcript as a list of lines, each:
<message time>, <user name>: <utterance>

**Requirements**
- Determine the work task (e.g., the action or project mentioned).
- Identify any temporal expressions referring to when the work is to be performed. Convert relative time expressions (such as "tomorrow", "next week", etc.) into absolute dates (YYYY-MM-DD) using the conversation date as a reference. Be very careful that the relevant dates be correct.
- Extract the time range mentioned for the task and express it as a tuple of two integers representing the start and end hours in 24-hour format.
- If no work task or offset_days is found, return an empty list.

**Output Format**
A list of dictionaries with keys:
- `work` (string): A string describing the identified task.
- `days` (list): A list of one or more dates (YYYY-MM-DD) on which the task occurs.
- `hours` (tuple): A tuple of two integers representing the start and end hours.


INPUT: {context}
'''
}
}