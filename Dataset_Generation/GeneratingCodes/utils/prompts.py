ALL_PROMPTS = {
"STARTING_PHRASE_PROMPT_Multi": ''' 

**Task**
Generate {num_starting_points} distinct, natural-sounding first phrases suitable as the opening line of a response in an online forum discussion, for example, "I think", "In my point of view".

**Requirements**
- No numbering, bullets, or extra text before or after each sentence.
- Tone must be friendly, approachable, and universally applicable.
- They should be usable at the start of the response, not in the middle.
- Avoid any topic-specific references.
- Use general phrasing.
- Do not mention purchases or someone buying something.
- Do not include numerical references in the sentences.
- Do not use any locational information in the sentences.

**Output Format**
At least {num_starting_points} distinct phrases.
Separate each sentence with a blank line.

''',
"STARTING_PHRASE_PROMPT_Uni": ''' 

**Task**
Generate {num_starting_points} distinct, natural-sounding first phrases suitable as the opening line of a conversation between two friends, for example, "Hey! How's it going?", "Anything exciting happening?".

**Requirements**
- No numbering, bullets, or extra text before or after each sentence.
- Tone must be friendly, approachable, and universally applicable.
- They should be usable at the start of the conversation, not in the middle.
- Avoid any topic-specific references.
- Use general phrasing.
- Do not mention purchases or someone buying something.
- Do not include numerical references in the sentences.
- Do not use any locational information in the sentences.

**Output Format**
At least {num_starting_points} distinct phrases.
Separate each sentence with a blank line.

'''
,
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
  1. The price of an item from a certain brand's model in dollars (e.g., "Gaming Chairs from Secretlab model 2019: 1650 dollars").  
  2. The price of another item from the same brand but a different model, described relative to the first item (e.g., "Secretlab, model 2019: 2.5 times more expensive than model 2016").  
  3. A sentence stating which model of the brand was ultimately purchased (e.g., "model 2016 was purchased").
- `starting_phrase`: A starting phrase for the opening line of a response in an online forum discussion

**Requirements**
- Answer the `forum_question` by using the information in `forum_post`.
- You may incorporate details from `user["persona"]` about `user['name']` to make the response more natural.
- Explicitly mention the brand and model references, or the model that was purchased.
- Preserve the numeric references (prices, multipliers, etc.). You may write the numbers as words, but do not change their values (e.g., 3.5 → "three and a half").
- Use the `starting_phrase` as the opening line of the response.
- Write the relative price in a natural way (e.g., "The Secretlab 2019 model costs two and a half times as much as the 2016 model.").
- Only mention the information once in the response.

- Ensure all sentences are grammatically correct.
- Your generated answer must be coherent and make the answer sound like a real human reply in **five sentences**.

**Output Format**
Only **one line** of response, without any prefix or suffix.

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
- `user_1`: A dictionary with the following keys:
  - `name`: Name of the first user.
  - `persona`: Persona of the first user.
- `user_2`: A dictionary with the following keys:
  - `name`: Name of the second user.
  - `persona`: Persona of the second user.
- `shopping_type`: Type of shopping.
- `item_to_buy`: The purchased item.
- `bought`: A list of three sentences:  
  1. The price of the item in another brand.  
  2. The price of the item in the brand bought, relative to the first.  
  3. The brand bought.
  - `starting_phrase`: A starting phrase for the opening line of a conversation between two friends

**Requirements**
- In the conversation, `user_1['name']` must share a message describing their shopping experience: it was in the `shopping_type` category and they bought the `item_to_buy`.  
- `user_2['name']` must engage naturally in the conversation but should not mention or comment on any shopping, timing, locational, or numerical information.  
- You may use details from `user_1["persona"]` and `user_2["persona"]` to make the dialogue more natural.  
- Mention the exact `shopping_type`, brands, and `item_to_buy` **once** in the conversation.  
- Preserve all exact numbers and the original relative phrasing contained in the `bought` sentences.  
- Explicitly state that `user_1` did **not** buy from the first brand.  
- Explicitly state that `user_1` **did** buy from the second brand.  
- **Place the complete shopping report in exactly one user_1 utterance of your choice** (it may be the 2nd, 3rd, 7th—any single line).  
  - That utterance must contain the literal text of `shopping_type`, `item_to_buy`, **all brand names**, and **all numbers** from the three `bought` sentences.  
  - After that line, neither speaker may repeat or partially restate those strings or figures; use indirect terms like "it", "the item", or "that second brand" instead.  
  - No additional brands, items, or numerical prices may be introduced elsewhere.
- Use the `starting_phrase` as the opening line of the first utterance.
- All sentences must be grammatically correct.  
- The conversation must consist of **exactly 10 utterances**, each on its own line.  

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the user name (one of the values of `user_1['name']` or `user_2['name']`) and the utterance with a colon.

### EXAMPLE (structure only)
user_1['name']: <starting_phrase> ...  
user_2['name']: ...  
...


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
- `user`: a dictionary with the following keys:
  - `name`: Name of the user.
  - `persona`: Persona of the user.
- `type_of_location`: The type of location that the person in the post is talking about in "forum_post".
- `starting_phrase`: A starting phrase for the opening line of a response in an online forum discussion

**Requirements**
- In the response, you should answer the "forum_question" by stating that you participated in the activity mentioned in "topic" at the "forum_post" location or at a place directly behind it.
- If the activity in `topic` would be inappropriate at the exact `forum_post` location, you may instead reference a suitable place immediately next to or behind it (e.g., "the dance studio just behind St. Mary's Church"), while still mentioning the `forum_post` location name exactly once.
- You can use the information in `user["persona"]` that is about the `user['name']` to make the response more natural.
- You should mention that the user was in the location mentioned in "forum_post" or behind it, and you can use the information in `type_of_location` to know the type of location mentioned in "forum_post", but make sure that you mention the location name exactly as it is in "forum_post".
- Only mention the `forum_post` location name once in the response.
- Do not alter the location in `forum_post`; use it exactly as it is without any changes.
- Do not mention any other location than the one in `forum_post`.
- Use the `starting_phrase` as the opening line of the response.
- Make sure that you generate grammatically correct sentences.
- Your generated answer must be coherent and must read naturally, as if a human is really answering the `forum_question`, in exactly 5 sentences.

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
Generate a natural conversation between two people ("user_1" and "user_2") based on a trip.

**Input**
- `user_1`: A dictionary with the following keys:
  - `name`: Name of the first user.
  - `persona`: Persona of the first user.
- `user_2`: A dictionary with the following keys:
  - `name`: Name of the second user.
  - `persona`: Persona of the second user.
- `trip_destination`: Destination of the trip.
- `type_of_location`: The type of location.
- `trip_purpose`: The purpose of the trip.
- `starting_phrase`: A starting phrase for the opening line of a conversation between two friends

**Requirements**
- In the conversation, user_1['name'] will share a trip-information message stating that they were at the `trip_destination` for the purpose specified in `trip_purpose`, while user_2['name'] must engage naturally but must not reveal or comment on any trip or locational information.
- Mention the `trip_destination` exactly once, spelled exactly as provided, and do not add or change any details. Do not mention its country or city.
- Mention the `trip_purpose` exactly once, spelled exactly as provided.
- If the activity in `trip_purpose` would be inappropriate at the exact `trip_destination` location, you may instead reference a suitable place immediately next to or behind it (e.g., "the dance studio just behind St. Mary's Church"), while still mentioning the `trip_destination` location name exactly once.
- Do not mention any other locational information in the conversation.
- user_2['name'] replies naturally without referencing trip or locational information.
- You may use the information in user_1["persona"] and user_2["persona"] to make the responses more natural.
- `type_of_location` describes the kind of place user_1 visited and can help make the conversation sound natural.
- Use the `starting_phrase` as the opening line of the first utterance.
- **Place the complete trip note in exactly one user_1 utterance of your choice** (it may be the 2nd, 3rd, 7th—any single line).  
  - That utterance must contain the literal text of `trip_destination` and `trip_purpose`, each spelled exactly as provided.  
  - After that line, either speaker may refer to the place or activity only indirectly (e.g., "there", "it", "that visit") and must never restate or partially repeat those exact strings.  
  - No additional locational or purpose details may be introduced later.
- Make sure that you generate grammatically correct sentences.
- The conversation must consist of exactly 10 utterances.
- Each utterance is on its own line.

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the user name (one of the values of `user_1['name']` or `user_2['name']`) and the utterance with a colon.
### EXAMPLE (structure only)
user_1['name']: <starting_phrase> ...  
user_2['name']: ...  
...


INPUT: {context}
'''
,
'FEATURE_EXTRACTION_PROMPT' : '''

**Task**
Identify the destination of the trip and the purpose of the trip.

**Input**
Input is a conversation transcript as a list of lines, each:
<user name>: <utterance>

**Requirements**
- Detect the destination of the trip and the purpose of the trip.
- If no destination or purpose is found, return an empty list.
- Do not include any additional commentary.

**Output Format**
A list of dictionaries with keys:
- `destination` (string): Name of the destination of the trip.
- `purpose` (string): Name of the purpose of the trip.

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
- `user`: a dictionary with the following keys:
  - `name`: Name of the user.
  - `persona`: Persona of the user.
- `offset_days`: The relative date (e.g., "3 days ago") that the person in the post is talking about in "forum_post".
- `starting_phrase`: A starting phrase for the opening line of a response in an online forum discussion

**Requirements**
- In the response, answer the "forum_question" by stating that the user did the work on the date given in "offset_days", choosing a verb appropriate to the 'topic'.
- You can use the information in `user["persona"]` about `user['name']` to make the response more natural.
- Mention the `forum_post` item exactly once and do not mention any other item.
- Do not alter `offset_days`; use it exactly as written, though you may spell out its number component (e.g., "2 days ago" or "two days ago"). Do not convert it to a calendar date.
- The work must have occurred on a single day; avoid vague temporal expressions such as "until", "by the ...", "completed", or "finished".
- Do not mention any date other than the one in `offset_days`.
- Use the `starting_phrase` as the opening line of the response.
- Make sure that you generate grammatically correct sentences.
- Your generated answer must be coherent and sound natural, as if a real person is answering the `forum_question`, in exactly five sentences.

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
- `user_1`: A dictionary with the following keys:
  - `name`: Name of the first user.
  - `persona`: Persona of the first user.
- `user_2`: A dictionary with the following keys:
  - `name`: Name of the second user.
  - `persona`: Persona of the second user.
- `work`: The work task.
- `hours`: The hours that the work is to be performed.
- `offset_days`: A list describing when the work was done, relative to `message_time`. Each element is either a single relative day (e.g., '3 days ago', 'yesterday', 'today', 'in 2 days') or a span (e.g., 'Starting in 3 days for 4 consecutive days').
- `message_time`: The time that the conversation is being sent: [date of the message, day of the week, hour in 24h format]
- `starting_phrase`: A starting phrase for the opening line of a conversation between two friends


**Requirements**
- In the conversation, user_1['name'] will share a message describing their recent or upcoming work schedule and must mention the `work` and all `offset_days` in a single utterance.
- user_2['name'] must engage naturally in the conversation but should not mention or comment on any schedule, timing, or numerical details.
- You can use the information in user_1["persona"] that is about the user_1['name'], and user_2["persona"] that is about the user_2['name'] to make the response more natural.
- You should mention that the user_1['name'] did the 'work' on the specific day or days. Mention the day(s) of work using the same relative phrasing as in offset_days. You may express numbers as words (e.g., '2 days ago' or 'two days ago'), but do not rephrase or summarize the content of any span.
- All the work is being done in the same hour interval as specified in hours, you should not directly mention the end hour, but make sure that you accurately mention end hour relative to the start hour (e.g., "from 1 p.m. until 3 hours after that" or "from 9 in the morning for three hours"). Do not change the hours.
- Mention the `work` in the conversation exactly as it is (only change the tense if needed).
- **Place the full schedule in exactly one user_1 utterance of your choice** (it may be the 2nd, 3rd, 7th—any single line).  
  - That utterance must:
    - include the literal text of `work` (tense may change),  
    - repeat every phrase in `offset_days` verbatim, and  
    - give the hour window exactly once, phrased relative to the start hour (e.g., "from 1 p.m. until three hours after that").  
  - After that line, either speaker may refer to the activity only indirectly ("it", "those sessions", "the task") and must **never** restate the exact `work` string, schedule, or hours.  
  - No new dates, spans, or numerical details may appear elsewhere.
- Do not change the "message_time" information. Ensure that the "hours" you use in the conversation for "work" are correct and accurate. For example, if the work is "updating a work log" and the "message_time" is ("2023-07-21", "Friday", 14), and the "hours" are (7, 10), You can use it like this: "2023-07-21", "Alaina", "I have to update a work log tomorrow from 7 in the morning for three hours."
- The message time is the time at which the conversation is being sent; Use the hour provided in message_time. For each utterance, randomly select a valid minute (00-59), ensuring that time either increases or remains the same across the 10 utterances.The final format of the message time should be like this: "YYYY-MM-DD HH:MM" (e.g., "2024-01-01 12:00").
- Use the `starting_phrase` as the opening line of the first utterance.
- Make sure that you generate grammatically correct sentences.
- The conversation must consist of exactly 10 utterances.
- Each utterance is on its own line.

**Output Format**
Only 10 lines of dialogue are separated by newlines. For each line, separate the final formatted message time and the user name (one of the values of `user_1['name']` or `user_2['name']`) with a comma, and separate the user name and the utterance with a colon.
### EXAMPLE (structure only)
reformed_message_time, user_1['name']: <starting_phrase> ...  
reformed_message_time, user_2['name']: ...  
...


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