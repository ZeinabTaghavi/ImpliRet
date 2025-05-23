import random
from datetime import timedelta, date, datetime
import ast
from pathlib import Path
import json
import os
# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------


data_human_evaluated = {
    35 : '''2020-03-02 10:05, Arleth: I have to monitor team activity, and I just realized I did it 16 days ago and 5 days ago, from 7 in the morning for three hours on both days.
2020-03-02 10:06, Nyalin: That sounds like a lot of work, how do you manage to keep everything organized?
2020-03-02 10:07, Arleth: It can be challenging, but I have a system in place that helps me stay on top of things, I'm just glad I was able to monitor team activity on those days.
2020-03-02 10:08, Nyalin: I'm sure it's not easy, but you seem to be doing a great job, what do you think is the most important part of monitoring team activity?
2020-03-02 10:09, Arleth: For me, it's about making sure everyone is working together seamlessly and that we're meeting our goals, it's a big responsibility but it's worth it in the end.
2020-03-02 10:10, Nyalin: That makes sense, teamwork is crucial in any project, have you noticed any improvements since you started monitoring team activity?
2020-03-02 10:11, Arleth: Yes, I have, and it's great to see everything coming together, I'm just taking it one day at a time and making sure I'm doing my best.
2020-03-02 10:12, Nyalin: That's a great attitude to have, it's always important to focus on the present moment and not get too caught up in the future or past.
2020-03-02 10:13, Nyalin: I'm curious, what do you like to do to unwind after a long day of monitoring team activity?
2020-03-02 10:14, Arleth: I like to take a walk and clear my head, it helps me relax and prepare for the next day, it's a great way to de-stress and recharge.''',
    121: '''2022-08-27 18:05, Karensa: I just wanted to let you know that I had to update the task board 24 days ago, and then again the day after that, and once more the day after that, all from 8 in the morning for three hours.
2022-08-27 18:07, Matilda: That sounds like a lot of work, how did it go?
2022-08-27 18:09, Karensa: It went well, but it was a bit tedious to update the task board for three consecutive days, I'm just glad it's done.
2022-08-27 18:11, Matilda: I can imagine, it's always a relief to finish a big task like that.
2022-08-27 18:13, Karensa: Definitely, and I'm happy to have had the opportunity to update the task board, even if it was a bit of a challenge.
2022-08-27 18:15, Matilda: You're always so diligent about keeping everything up to date, it's really impressive.
2022-08-27 18:17, Karensa: Thanks, I just like to make sure everything is organized and running smoothly, it makes life easier in the long run.
2022-08-27 18:19, Matilda: That's a great attitude to have, it's always better to stay on top of things.
2022-08-27 18:21, Karensa: Exactly, and I'm just glad that I could update the task board when I did, it's nice to have it out of the way.
2022-08-27 18:23, Matilda: I'm sure it's a weight off your shoulders, now you can focus on other things.''',
    192: '''2021-07-24 10:05, Dianthe: I just wanted to share with you that I had to prepare a meal 24 days ago from 12 in the afternoon for 2 hours.
2021-07-24 10:07, Kajari: That sounds like quite a task, how did it go?
2021-07-24 10:10, Dianthe: It was a bit challenging, but I managed to get everything done on time, thankfully it was a one-time thing.
2021-07-24 10:12, Kajari: I'm glad to hear that, what kind of meal were you preparing?
2021-07-24 10:15, Dianthe: It was a pretty standard meal, but I had to make sure everything was perfect, so it took some time and effort.
2021-07-24 10:18, Kajari: I can imagine, attention to detail is important when it comes to cooking.
2021-07-24 10:20, Dianthe: Exactly, and I'm just glad that it's done and I don't have to worry about it anymore.
2021-07-24 10:22, Kajari: Well, if you ever need any help or just want to cook together, let me know.
2021-07-24 10:25, Dianthe: Thanks for the offer, I might take you up on that sometime, it could be fun to cook together.
2021-07-24 10:27, Kajari: Sounds great, I'm looking forward to it, have a great day!''',
    611 : '''2023-06-10 09:05, Maebry: I just wanted to let you know that I updated a project report 24 days ago from 7 in the morning for 4 hours.
2023-06-10 09:07, Odessa: That sounds like a lot of work, how did it go?
2023-06-10 09:10, Maebry: It was a bit challenging, but I was able to finish it on time, the work was done in one day, as it was a one-time task.
2023-06-10 09:12, Odessa: I'm glad to hear that, you must be relieved that it's over.
2023-06-10 09:15, Maebry: Yes, I am, it was a big task, but now that it's done, I can focus on other things, and I'm looking forward to my next project.
2023-06-10 09:18, Odessa: That's great, do you have any fun plans for the rest of the day?
2023-06-10 09:20, Maebry: Not really, just some errands to run, but I'm hoping to have some time to relax later, maybe catch up on some reading.
2023-06-10 09:22, Odessa: That sounds like a good way to unwind, you deserve it after finishing that big project.
2023-06-10 09:25, Maebry: Thanks, I'm just glad that the update a project report task is behind me now, and I can move on to other things.
2023-06-10 09:28, Odessa: Well, I'm sure you'll do great on your next project, you're always so organized and focused.''', 
    640: '''2020-01-29 18:05, Alaina: I just wanted to let you know that I reviewed a performance report 24 days ago from 3 in the afternoon for 2 hours.
2020-01-29 18:07, Cypress: That sounds like a lot of work, how did it go?
2020-01-29 18:10, Alaina: It was a lot to take in, but I think I got a good understanding of what needs to be improved.
2020-01-29 18:12, Cypress: I'm sure it's not easy to review those kinds of reports, but it's great that you're on top of it.
2020-01-29 18:15, Alaina: Yeah, it's definitely important to stay on top of it, and I'm just glad I could get it done 24 days ago.
2020-01-29 18:17, Cypress: I'm sure it's a big relief to have it off your plate, what's next for you?
2020-01-29 18:20, Alaina: I've got a few other things I need to take care of, but I'm feeling pretty caught up after finishing the report.
2020-01-29 18:22, Cypress: That's great to hear, you always seem to be so organized and on top of things.
2020-01-29 18:25, Alaina: Thanks, I try my best to stay organized, it's just part of the job, and I'm glad I could review the performance report when I did.
2020-01-29 18:27, Cypress: Well, you're doing a great job, keep up the good work!''', 
    746: '''2023-10-07 13:05, Odessa: I have to compile a weekly summary 20 days later from 9 in the morning for 2 hours.
2023-10-07 13:07, Bastien: That sounds like a lot of work, are you prepared for it?
2023-10-07 13:10, Odessa: Yeah, I've been gathering all the necessary information, so I'm feeling pretty confident about it.
2023-10-07 13:12, Bastien: That's great to hear, I'm sure you'll do a fantastic job.
2023-10-07 13:15, Odessa: Thanks for the vote of confidence, I really appreciate it.
2023-10-07 13:17, Bastien: So, what made you think about the weekly summary today?
2023-10-07 13:20, Odessa: I just wanted to make sure I had enough time to get everything done, you know how it is.
2023-10-07 13:22, Bastien: Yeah, it's always a good idea to plan ahead and make sure you're on track.
2023-10-07 13:25, Odessa: Exactly, and I'm hoping that by doing it 20 days later, I'll have all the information I need to make it really comprehensive.
2023-10-07 13:27, Bastien: I'm sure it'll be fine, just take your time and do your best, that's all anyone can ask for.''', 
    1071: '''2023-01-28 15:10, Lilou: I have to revise a project timeline 25 days later from 9 in the morning for two hours.
2023-01-28 15:11, Bellamy: That sounds like a big task, how are you preparing for it?
2023-01-28 15:12, Lilou: I'm just making sure I have all the necessary materials and information to get it done efficiently.
2023-01-28 15:13, Bellamy: It's great that you're being thorough, I'm sure it will pay off in the end.
2023-01-28 15:14, Lilou: Yeah, I hope so, I just want to make sure everything is perfect.
2023-01-28 15:15, Bellamy: I'm sure you'll do great, you're very detail-oriented.
2023-01-28 15:16, Lilou: Thanks, I try my best to make sure everything is just right.
2023-01-28 15:17, Bellamy: So, what do you like to do to relax after working on a big project?
2023-01-28 15:18, Lilou: I usually like to take a walk or read a book, it helps me unwind.
2023-01-28 15:19, Bellamy: That sounds lovely, I might have to try that sometime.''', 
    1131: '''2024-10-01 10:05, Veronica: I have to draft a customer feedback report 23 days later from 7 in the morning for three hours.
2024-10-01 10:06, Fortune: That sounds like a significant task, how are you preparing for it?
2024-10-01 10:07, Veronica: I've been gathering all the necessary data and feedback from our customers, it's been a lengthy process.
2024-10-01 10:08, Fortune: I can imagine, but it's great that you're taking the time to get everything just right.
2024-10-01 10:09, Veronica: Yes, accuracy is key when it comes to this type of report, I want to make sure it's perfect.
2024-10-01 10:10, Fortune: I'm sure it will be, you're very detailed-oriented and thorough in your work.
2024-10-01 10:11, Veronica: Thanks, I try my best, and I'm hoping this report will be helpful in improving our customer service.
2024-10-01 10:12, Fortune: I'm sure it will be, have you thought about how you'll be presenting the findings?
2024-10-01 10:13, Veronica: Not yet, but that's something I'll be considering after I've completed the report.
2024-10-01 10:14, Fortune: Well, if you need any help or just want to bounce ideas off me, I'm here to listen.''', 
    1170: '''2024-03-12 08:05, Radella: I have to review daily feedback from 3 in the afternoon for 4 hours, and I've been doing it for the past few days, specifically 15 days ago, 14 days ago, and 13 days ago.
2024-03-12 08:07, Kalina: That sounds like a lot of work, how have you been managing it?
2024-03-12 08:10, Radella: It's been a challenge, but I'm trying to stay on top of it, and I've been focusing on getting it done during the scheduled hours.
2024-03-12 08:12, Kalina: I'm sure it's not easy, but you seem to be handling it well, what's your strategy for staying organized?
2024-03-12 08:15, Radella: Well, I just try to prioritize my tasks and make sure I have enough time for everything, and it's working out so far.
2024-03-12 08:18, Kalina: That makes sense, prioritizing tasks is always a good idea, do you think you'll be able to keep up this pace?
2024-03-12 08:20, Radella: I hope so, I'm trying to stay consistent and not fall behind, but it's hard to say for sure.
2024-03-12 08:22, Kalina: I'm sure you'll be fine, you're very capable, and it's great that you're being proactive about it.
2024-03-12 08:25, Kalina: Is there anything I can do to help or support you in any way?
2024-03-12 08:28, Radella: Actually, just talking about it helps, thanks for listening, Kalina.''', 
    1174: '''2024-03-18 17:05, Radella: I just wanted to let you know that I have been reading a technical article 21 days ago, 20 days ago, and 10 days ago from 11 in the morning for 2 hours.
2024-03-18 17:07, Bellamy: What's the article about, is it interesting?
2024-03-18 17:10, Radella: Yes, it's very informative, I'm glad I had the chance to read it on those days.
2024-03-18 17:12, Bellamy: That sounds great, I'm sure you learned a lot from it.
2024-03-18 17:15, Radella: Definitely, it was worth taking the time to read it.
2024-03-18 17:18, Bellamy: I'm happy to hear that, it's always good to expand our knowledge.
2024-03-18 17:20, Radella: Absolutely, I feel like I have a better understanding of the topic now.
2024-03-18 17:22, Bellamy: That's terrific, I'm sure it will be helpful in the future.
2024-03-18 17:25, Radella: Yeah, I'm looking forward to applying what I learned.
2024-03-18 17:28, Bellamy: Well, if you have any questions or need any help, feel free to ask me anytime.''',
    1175: '''2024-03-22 17:05, Radella: I just wanted to share that I had to attend daily stand-up meeting 24 days ago and 23 days ago from 8 in the morning for two hours.
2024-03-22 17:07, Elita: That sounds like it was a busy time, how did the meetings go?
2024-03-22 17:10, Radella: Yeah, they were pretty productive, we were able to discuss and resolve some important issues.
2024-03-22 17:12, Elita: I'm glad to hear that, it's always great when meetings can be useful and efficient.
2024-03-22 17:15, Radella: Definitely, it's always a good feeling when you can come out of a meeting feeling like you've accomplished something.
2024-03-22 17:18, Elita: So, what were some of the key topics that you discussed during the meetings?
2024-03-22 17:21, Radella: We talked about some new projects that are coming up and how we can work together to make them successful.
2024-03-22 17:24, Elita: That sounds exciting, I'm sure it'll be a great opportunity for growth and learning.
2024-03-22 17:27, Radella: Yeah, I'm looking forward to seeing how everything comes together, it should be a interesting few weeks.
2024-03-22 17:30, Elita: I'm sure you'll do great, you always seem to handle these types of situations with ease.''', 
    1179: '''2024-03-10 10:05, Radella: I have to tell you that I completed an expense report 13 days ago from 7 in the morning for three hours.
2024-03-10 10:07, Cypress: That sounds like a lot of work, how did it go?
2024-03-10 10:09, Radella: It was a bit tedious, but I'm just glad it's done and I can focus on other things now.
2024-03-10 10:11, Cypress: I can imagine, it's always a relief to finish a big task like that.
2024-03-10 10:13, Radella: Exactly, now I can move on to more important things and not have that hanging over my head.
2024-03-10 10:15, Cypress: That's great, you must be feeling really productive now.
2024-03-10 10:17, Radella: Yeah, I am, it's always nice to feel like I'm getting things done and making progress.
2024-03-10 10:19, Cypress: I'm sure it's not easy, but you seem to handle it all really well.
2024-03-10 10:21, Radella: Thanks, I try my best to stay organized and manage my time effectively.
2024-03-10 10:23, Cypress: Well, it definitely seems like you're doing something right, keep up the good work.''', 
    1397: '''2022-09-06 08:05, Keanu: I have to update a work log from 17 in the evening for 2 hours, starting 17 days later, and then for the next 3 days.
2022-09-06 08:07, Lilou: That sounds like a lot of work, are you feeling overwhelmed?
2022-09-06 08:10, Keanu: Yeah, a bit, I have to make sure everything is up to date, so I'll be updating a work log from 17 in the evening for 2 hours on each of those days.
2022-09-06 08:12, Lilou: I'm sure you'll manage, you're very organized, what's the plan for the updates?
2022-09-06 08:15, Keanu: Well, I'll be doing it at the same time every day, from 17 in the evening for 2 hours, 17 days later, then 18 days later, 19 days later, and finally 20 days later.
2022-09-06 08:18, Lilou: That sounds like a good strategy, do you think it will take longer on any of the days?
2022-09-06 08:20, Keanu: I don't think so, I'll be updating a work log for 2 hours on each day, starting 17 days later, and it should be the same amount of work each time.
2022-09-06 08:22, Lilou: Okay, that's good, I hope everything goes smoothly, do you need any help?
2022-09-06 08:25, Keanu: No, I don't think so, but thanks for offering, I'll be updating a work log from 17 in the evening for 2 hours, and I should be able to handle it.
2022-09-06 08:28, Lilou: Alright, if you're sure, I'll let you get back to work then, good luck with the updates.''', 
    1468: '''2020-11-25 16:05, Clarion: I just wanted to let you know that I will be updating a project report 23 days later from 7 in the morning for three hours.
2020-11-25 16:07, Ashriel: That sounds like a significant task, how are you preparing for it?
2020-11-25 16:10, Clarion: I've been going over the project details and making sure I have all the necessary information to include in the report.
2020-11-25 16:12, Ashriel: It's great that you're being thorough, I'm sure it will pay off in the end.
2020-11-25 16:15, Clarion: Thanks, I'm hoping it will all come together smoothly, I'll be sure to let you know how it goes.
2020-11-25 16:17, Ashriel: I appreciate that, I'm looking forward to hearing about your progress.
2020-11-25 16:20, Clarion: I'm expecting it to be a bit challenging, but I'm confident that I can get it done.
2020-11-25 16:22, Ashriel: You've handled tough tasks before, I'm sure this will be no exception.
2020-11-25 16:25, Clarion: Thanks for the vote of confidence, it means a lot to me.
2020-11-25 16:27, Ashriel: Anytime, that's what friends are for, good luck with everything.'''
}
random.seed(42)

def merging_dataset(base_path):

    # Load the structured data
    structured_data_path = base_path + "Dataset_Helping/T_Uni/T_Uni_Structured.jsonl"

    with open(structured_data_path, 'r', encoding='utf-8') as f:
        structured_data = [json.loads(line) for line in f]

    print(f"Loaded {len(structured_data)} records from structured data file")

    # Load the generated conversation data
    generated_data_path = base_path + "Dataset_Helping/T_Uni/T_Uni_Structured_Generated_conversation.jsonl"

    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = [json.loads(line) for line in f]

    print(f"Loaded {len(generated_data)} records from generated data file")

    dataset = []

    error_list = [35, 121, 192, 611, 640, 746, 1071, 1131, 1170, 1174, 1175, 1179, 1397, 1468]

    x = 0
    for i in range(len(structured_data)):
        
        user_ID = i
        user = structured_data[i]['user_1']
        schedule_info_list = structured_data[i]['schedule']

        for j in range(len(schedule_info_list)):
            user_2 = schedule_info_list[j]['user_2']
            
            assert i*30 + j == x
            x += 1

            conversation = generated_data[int(i*30 + j)].replace(f"{user}: {user_2}:", f"{user_2}:").replace(f"{user_2}: {user}:", f"{user}:").replace(f"{user_2}: {user_2}:", f"{user_2}:").replace(f"{user}: {user}:", f"{user}:")


            if i*30 + j in error_list:
                conversation = data_human_evaluated[int(i*30 + j)]

            # user_response = generated_data[int(i*20 + j)]
            if conversation == '-':
                print(conversation, i*20 + j)
                raise Exception("conversation is '-'")
            work_date = datetime.strptime(schedule_info_list[j]['question_time'][0], "%Y-%m-%d").strftime("%B %d, %Y")
            work_hour = schedule_info_list[j]['question_time'][1]
            question = f"What was {user} scheduled to be doing at {work_hour}:00 on {work_date}?"
            dataset.append({
                "user_ID": user_ID,
                "user": user,
                "user_2": user_2,
                "context": conversation,
                "extra_info": {k:v for k,v in schedule_info_list[j].items() if k in ['activity_type', 'days', 'hours']},
                "question": question,
                "answer": schedule_info_list[j]['work']
            })

    with open(base_path + '/Data/T_Uni.jsonl', 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"T_Uni: {len(dataset)}, stored in {base_path}/Data/T_Uni.jsonl")
if __name__ == "__main__":
    base_path = './Dataset_Generation/'
    merging_dataset(base_path)