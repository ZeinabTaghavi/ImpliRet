
from huggingface_hub import login
from vllm import LLM, SamplingParams
from utils.prompts import ALL_PROMPTS
from utils.loading_model import load_model
from utils.feature_extraction import *
import json
import argparse
import random
import re
import os
import torch
import ast

# ------------------------------------------------------------
# Initialize random seed
# ------------------------------------------------------------

random.seed(42)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def load_jsonl(filename):
    try:
        with open(filename, 'r') as f:
            return [json.loads(line) for line in f]
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

def filter_generated_conversation_responses(generated_conversation, num_responses, return_type = 'string', validation_type = 'more_than_ok'):
    mistaken_conversation_idx = []
    conversation_list = []

    for idx, text in enumerate(generated_conversation):
        # Normalise newlines and split on blank space
        if type(text) == list:
            tokens = [t for t in text if t.strip() != '']
        else:
            tokens = [
                tok for tok in text.replace('\\n', '\n').split('\n')
                if (tok.strip() != '' and tok.strip() != '-')
            ]

        if len(tokens) == num_responses or (validation_type == 'more_than_ok' and len(tokens) > num_responses):
            if return_type == 'string':
                # Accept as‑is (you can post‑process further if desired)
                conversation_list.append(text)
            elif return_type == 'list':
                conversation_list.append(tokens[:num_responses])
        else:
            # print(tokens)
            # print(len(tokens))
            # print('----------------------------')
            # Invalid – keep placeholder and record index
            conversation_list.append('-')
            mistaken_conversation_idx.append(idx)

    print('len(conversation_list)', len(conversation_list))
    return mistaken_conversation_idx, conversation_list

    

def several_attempts_generation(evaluation_function,
                               llm,
                               prompts: list[list[dict]],
                               total_attempts: int,
                               num_expected_lines: int,
                               temperature: float = 1.0,
                               llm_outputs: list[str] = None,
                               llm_output_prepared: bool = False,
                               max_tokens: int = 4096,
                               output_type: str = 'list',
                               more_than_ok_type: str = 'more_than_ok'):

    if not llm_output_prepared:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = llm.chat(messages=prompts,
                        sampling_params=sampling_params,
                        use_tqdm=True)
        llm_outputs = [o.outputs[0].text for o in outputs]
    else:
        llm_outputs = llm_outputs

    # Attempt to fix conversations
    for attempt in range(total_attempts):
        print(f"Soft Attempt {attempt} of {total_attempts}")
        mistaken_idx, llm_outputs = evaluation_function(llm_outputs, num_expected_lines, output_type, more_than_ok_type)
        print(f"Number of mistaken generated outputs: {len(mistaken_idx)}")
        print(f"Number of generated outputs: {len(llm_outputs)}")
        if len(mistaken_idx) == 0:
            print("No mistaken generated outputs")
            break
        else:
            mistaken_prompts = [prompts[i] for i in mistaken_idx]
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = llm.chat(messages=mistaken_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
            re_generated_llm_outputs = [o.outputs[0].text for o in outputs]
            for i in range(len(mistaken_idx)):
                llm_outputs[mistaken_idx[i]] = re_generated_llm_outputs[i]
    
    mistaken_idx, llm_outputs = evaluation_function(llm_outputs, num_expected_lines, output_type, more_than_ok_type)
    
    print(f"Number of mistaken generated outputs: {len(mistaken_idx)}")
    print(f"Number of generated outputs: {len(llm_outputs)}")

    return mistaken_idx, llm_outputs

def experiment_setup(num_gpus: int, 
                 model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
                 track: str = 'A',
                 conv_type: str = 'Multi',
                 datasets_helping_folder: str = './Dataset_Generation/Dataset_Helping',
                 dataset_bunch_key: str = 'shopping_info'):


    # Load dataset and generate outputs
    dataset = load_jsonl(f'{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured.jsonl')
    output_filename_starting_main_conversations = f"{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured_Generated_starting_main_conversations.jsonl"
    output_filename_conversation = f"{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured_Generated_conversation.jsonl"
    output_filename_feature_extraction = f"{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured_Generated_feature_extraction.jsonl"
    output_filename_extra_samples = f"{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured_Generated_extra_samples.jsonl"
    output_filename_starting_conversations = f"{datasets_helping_folder}/{track}_{conv_type}/{track}_{conv_type}_Structured_Generated_starting_conversations.jsonl"

    if track == 'A' and conv_type == 'Multi':
        dataset_bunch_key = 'posts'
        selected_info_keys = ['forum_post', 'question'] # for feature extraction
        conversation_generation_keys_main = ['topic', 'forum_question'] # for conversation generation prompt
        conversation_generation_keys = ['forum_post_2']
        
        feature_extraction_function = filter_generated_feature_parsing_step_A_Multi

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_A_Multi']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_A_Multi']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_A_Multi']['FEATURE_EXTRACTION_PROMPT']

    elif track == 'A' and conv_type == 'Uni':
        dataset_bunch_key = 'shopping_info'
        selected_info_keys = ['shopping_type', 'item_to_buy' , 'bought', 'final_price']
        conversation_generation_keys_main = ['user_1']
        conversation_generation_keys = ['user_2', 'shopping_type', 'item_to_buy', 'bought']

        feature_extraction_function = filter_generated_feature_parsing_step_A_Uni 

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_A_Uni']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_A_Uni']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_A_Uni']['FEATURE_EXTRACTION_PROMPT']

    elif track == 'S' and conv_type == 'Multi':
        dataset_bunch_key = 'posts'
        selected_info_keys = ['forum_post', 'question'] # for feature extraction
        conversation_generation_keys_main = ['topic', 'forum_question'] # for conversation generation prompt
        conversation_generation_keys = ['forum_post', 'destination_type'] 
        
        feature_extraction_function = filter_generated_feature_parsing_step_S_Multi

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_S_Multi']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_S_Multi']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_S_Multi']['FEATURE_EXTRACTION_PROMPT']

    elif track == 'S' and conv_type == 'Uni':
        dataset_bunch_key = 'trip_info'
        selected_info_keys = ['trip_destination', 'trip_friends']
        conversation_generation_keys_main = ['user_1']
        conversation_generation_keys = ['user_2', 'trip_destination', 'type_of_location', 'trip_friends']

        feature_extraction_function = filter_generated_feature_parsing_step_S_Uni

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_S_Uni']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_S_Uni']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_S_Uni']['FEATURE_EXTRACTION_PROMPT']
    
    elif track == 'T' and conv_type == 'Multi':
        dataset_bunch_key = 'posts'
        selected_info_keys = ['offset_days'] # for feature extraction
        conversation_generation_keys_main = ['topic', 'forum_question'] # for conversation generation prompt
        conversation_generation_keys = ['forum_post', 'offset_days']
        
        feature_extraction_function = filter_generated_works_parsing_step_T_Multi

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_T_Multi']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_T_Multi']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_T_Multi']['FEATURE_EXTRACTION_PROMPT']

    elif track == 'T' and conv_type == 'Uni':
        dataset_bunch_key = 'schedule'
        selected_info_keys = ['days', 'hours']
        conversation_generation_keys_main = ['user_1']
        conversation_generation_keys = ['user_2', 'work', 'activity_type', 'hours', 'offset_days', 'message_time']

        feature_extraction_function = filter_generated_works_parsing_step_T_Uni

        STARTING_CONVERSATION_PROMPT = ALL_PROMPTS['PROMPTS_T_Uni']['STARTING_CONVERSATION_PROMPT']
        CONVERSATION_GENERATION_PROMPT = ALL_PROMPTS['PROMPTS_T_Uni']['CONVERSATION_GENERATION_PROMPT']
        FEATURE_EXTRACTION_PROMPT = ALL_PROMPTS['PROMPTS_T_Uni']['FEATURE_EXTRACTION_PROMPT']

    if conv_type == 'Multi':
        q_num_per_session = 20
        num_sessions = 25
        conv_lines = 1
    elif conv_type == 'Uni':
        q_num_per_session = 30
        num_sessions = 50
        conv_lines = 10
    
    assert len(dataset) == num_sessions, f"Number of sessions in the dataset is not equal to the number of sessions in the experiment. {len(dataset)} != {num_sessions}"
    assert len(dataset[0][dataset_bunch_key]) == q_num_per_session, f"Number of questions per session in the dataset is not equal to the number of questions per session in the experiment. {len(dataset[0][dataset_bunch_key])} != {q_num_per_session}"

    

    experiment = {
        'track': track,
        'conv_type': conv_type,
        'dataset': dataset,
        'q_num_per_session': q_num_per_session,
        'num_sessions': num_sessions,
        'output_filename_conversation': output_filename_conversation,
        'output_filename_feature_extraction': output_filename_feature_extraction,
        'output_filename_extra_samples': output_filename_extra_samples,
        'output_filename_starting_conversations': output_filename_starting_conversations,
        'output_filename_starting_main_conversations': output_filename_starting_main_conversations,
        'dataset_bunch_key': dataset_bunch_key,
        'datasets_helping_folder': datasets_helping_folder,
        'selected_info_keys': selected_info_keys,
        'conversation_generation_keys_main': conversation_generation_keys_main,
        'conversation_generation_keys': conversation_generation_keys,
        'conv_lines': conv_lines,
        'feature_extraction_function': feature_extraction_function,
        'STARTING_CONVERSATION_PROMPT': STARTING_CONVERSATION_PROMPT,
        'CONVERSATION_GENERATION_PROMPT': CONVERSATION_GENERATION_PROMPT,
        'FEATURE_EXTRACTION_PROMPT': FEATURE_EXTRACTION_PROMPT,
    }
    return experiment

def main(num_gpus: int = 4, 
         model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
         track: str = 'A',
         conv_type: str = 'Uni',
         datasets_helping_folder: str = './Dataset_Generation/Dataset_Helping',
         temperature: float = 1,
         total_attempts: int = 35,
         total_attempts_hard: int = 20,
         max_tokens_starting_conv: int = 4096,
         max_tokens_conversation_generation: int = 1024,
         max_tokens_feature_extraction: int = 512,):

    """Main execution function."""
    # Initialize model
    print("Loading the model...")
    llm = load_model(model_name, num_gpus)
    
    experiment = experiment_setup(num_gpus, model_name, track, conv_type, datasets_helping_folder)
   
    print('-------------------------------- Generating starting lines of conversations --------------------------------')
    if os.path.exists(experiment['output_filename_starting_main_conversations']):
        print(f"Loading starting conversations from {experiment['output_filename_starting_main_conversations']}")
        starting_conversations = load_jsonl(experiment['output_filename_starting_main_conversations'])
    else:
        prompts_starting_conv = [ [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": experiment['STARTING_CONVERSATION_PROMPT'].format(num_starting_points=experiment['q_num_per_session'])
                }
                
            ] for _ in range(int(experiment['num_sessions']))]

        print(f'Number of itteration for generating {experiment["q_num_per_session"]} strarting points: {len((prompts_starting_conv))}')
        _, starting_conversations = several_attempts_generation(total_attempts=total_attempts,
                                                            temperature=temperature,
                                                            llm=llm,
                                                            evaluation_function=filter_generated_conversation_responses,
                                                            llm_output_prepared=False,
                                                            prompts=prompts_starting_conv,
                                                            num_expected_lines=experiment['q_num_per_session'],
                                                            max_tokens=max_tokens_starting_conv,
                                                            output_type='list',
                                                            more_than_ok_type='more_than_ok')

        print(f"saving starting conversations...")
        with open(experiment['output_filename_starting_main_conversations'], 'w') as f:
            for output in starting_conversations:
                json.dump(output, f)
                f.write('\n')
        print(f"starting conversations saved to {experiment['output_filename_starting_main_conversations']}")
    
    print('---------------------------------------- Generating the conversations ----------------------------------------')
    # Prepare prompts for conversation generation
    prompts_conversation_generation = []
    selected_info_list = []
    for i in range(len(experiment['dataset'])):
        if experiment['conv_type'] == 'Uni':
            user = experiment['dataset'][i]['user_1']
        for i_2, conversation_bunch in enumerate(experiment['dataset'][i][experiment['dataset_bunch_key']]):
            starting_conv = starting_conversations[i][i_2]
            selected_info = {key: conversation_bunch[key] for key in experiment['selected_info_keys']}
            input_dict = {}

            # if you need to add more keys to the input_dict, you can do it here
            # it is adding from the data row
            for key in experiment['conversation_generation_keys_main']:
                input_dict[key] = experiment['dataset'][i][key]

            # this is adding from the conversation_bunch
            for key in experiment['conversation_generation_keys']:
                try:
                    second_key = int(key.split('_')[-1])
                    first_key = '_'.join(key.split('_')[:-1])
                    input_dict[first_key] = conversation_bunch[first_key][int(second_key)]
                except:
                    input_dict[key] = conversation_bunch[key]

            input_dict['starting_conv'] = starting_conv
            conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates a conversation between two people based on a shopping list."
                    },
                    {
                        "role": "user",
                        "content": experiment['CONVERSATION_GENERATION_PROMPT'].format(context=input_dict)
                    }
                ]

            selected_info_list.append(selected_info)
            prompts_conversation_generation.append(conversation)                                                         
 

    print('length of prompts_conversation_generation: ', len(prompts_conversation_generation))
    
    if not os.path.exists(experiment['output_filename_conversation']):
        print("Generating outputs in step 1...")
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_conversation_generation)
        outputs = llm.chat(messages=prompts_conversation_generation,
                    sampling_params=sampling_params,
                    use_tqdm=True)

        llm_outputs_conversation = [o.outputs[0].text for o in outputs]
        llm_outputs_feature_extraction = ['-' for o in outputs]

        mistaken_conversation_idx, conversation_list = filter_generated_conversation_responses(llm_outputs_conversation, experiment['conv_lines'], 'string', 'more_than_not_ok')
        mistaken_extracted_idx, extracted_feature_list = experiment['feature_extraction_function'](llm_outputs_feature_extraction, selected_info_list)
        print("End of step 1")
    else:
        print("Loading outputs from step 1...")
        llm_outputs_conversation = load_jsonl(experiment['output_filename_conversation'])
        llm_outputs_feature_extraction = load_jsonl(experiment['output_filename_feature_extraction'])

        mistaken_conversation_idx, conversation_list = filter_generated_conversation_responses(llm_outputs_conversation, experiment['conv_lines'], 'string', 'more_than_not_ok')
        mistaken_extracted_idx, extracted_feature_list = experiment['feature_extraction_function'](llm_outputs_feature_extraction, selected_info_list)

        print(f"Outputs loaded from {experiment['output_filename_conversation']} and {experiment['output_filename_feature_extraction']}")
        print(f"Number of mistaken conversations: {len(mistaken_conversation_idx)}")
        print(f"Number of mistaken extractions: {len(mistaken_extracted_idx)}")
        print("End of step 1")

    print(f"mistaken_extracted_idx: {mistaken_extracted_idx}")
    if len(mistaken_extracted_idx) > 0 or len(mistaken_conversation_idx) > 0:
        for t_a_hard in range(total_attempts_hard):
            print(f'---------------------------------------- Hard Attempt {t_a_hard} of {total_attempts_hard} ----------------------------------------')
            mistaken_conversation_idx, conversation_list = several_attempts_generation(total_attempts=total_attempts,
                                                            temperature=temperature,
                                                            llm=llm,
                                                            evaluation_function=filter_generated_conversation_responses,
                                                            llm_output_prepared=True,
                                                            llm_outputs=conversation_list,
                                                            prompts=prompts_conversation_generation,
                                                            num_expected_lines=experiment['conv_lines'],
                                                            max_tokens=max_tokens_conversation_generation,
                                                            output_type='string',
                                                            more_than_ok_type='more_than_not_ok')

            print(f"Number of mistaken generated conversations: {len(mistaken_conversation_idx)}")
            for i in mistaken_conversation_idx:
                print(f"Mistaken conversation: {conversation_list[i]}")
            print(f"Number of generated conversations: {len(conversation_list)}")

            # Generate feature extraction prompts
            prompts_feature_extraction = []
            for i in range(len(conversation_list)):
                conversation = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that extracts features from a conversation between two people."
                        },
                        {
                            "role": "user",
                            "content": experiment['FEATURE_EXTRACTION_PROMPT'].format(context=conversation_list[i])
                        }
                    ]
                prompts_feature_extraction.append(conversation)
            
            # Generate feature extractions
            print("Generating extractions...")
            print(f"1 - Number of extractions that will be generated: {len(mistaken_extracted_idx)}")
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_feature_extraction)
            selected_prompts = [prompts_feature_extraction[i] for i in mistaken_extracted_idx]
            outputs = llm.chat(messages=selected_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=True)
            for i in range(len(mistaken_extracted_idx)):
                llm_outputs_feature_extraction[mistaken_extracted_idx[i]] = outputs[i].outputs[0].text

            # Attempt to fix feature extractions
            print("2 - Regenerating works if needed")
            for a_t_soft in range(total_attempts):
                print(f"Regenerating works: Attempt {a_t_soft} of {total_attempts}")
                mistaken_extracted_idx, extracted_feature_list = experiment['feature_extraction_function'](llm_outputs_feature_extraction, selected_info_list)

                if len(mistaken_extracted_idx) == 0:
                    break
                else:
                    mistaken_prompts = [prompts_feature_extraction[i] for i in mistaken_extracted_idx]
                    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_feature_extraction)
                    outputs = llm.chat(messages=mistaken_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=True)
                    re_generated_llm_outputs = [o.outputs[0].text for o in outputs]
                    for i in range(len(mistaken_extracted_idx)):
                        llm_outputs_feature_extraction[mistaken_extracted_idx[i]] = re_generated_llm_outputs[i]

            # Final validation and regeneration if needed
            mistaken_extracted_idx, extracted_feature_list = experiment['feature_extraction_function'](llm_outputs_feature_extraction, selected_info_list)
            print(f"Number of mistaken works in after {total_attempts_hard} attempts: {len(mistaken_extracted_idx)}")
            print("3 - Regenerating conversation if needed...")
            if len(mistaken_extracted_idx) != 0:
                print(f"Number of mistaken works in after {total_attempts} attempts: {len(mistaken_extracted_idx)}")
                # Generate new conversations
                new_conv_prompts = [prompts_conversation_generation[i] for i in mistaken_extracted_idx]
                sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_conversation_generation)
                outputs = llm.chat(messages=new_conv_prompts,
                        sampling_params=sampling_params,
                        use_tqdm=True)
                re_generated_llm_outputs = [o.outputs[0].text for o in outputs]
                for i in range(len(mistaken_extracted_idx)):
                    llm_outputs_conversation[mistaken_extracted_idx[i]] = re_generated_llm_outputs[i]
                
                # Generate new feature extractions
                mistaken_prompts = [[ { "role": "system",
                                        "content": "You are a helpful assistant that extracts features from a conversation between two people."
                                    },{
                                        "role": "user",
                                        "content": experiment['FEATURE_EXTRACTION_PROMPT'].format(context=re_generated_llm_outputs[i])
                                    }] for i in range(len(re_generated_llm_outputs))]
                sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens_feature_extraction)
                outputs = llm.chat(messages=mistaken_prompts,
                        sampling_params=sampling_params,
                        use_tqdm=True)
                for i in range(len(mistaken_extracted_idx)):
                    llm_outputs_feature_extraction[mistaken_extracted_idx[i]] = outputs[i].outputs[0].text

                mistaken_extracted_idx, extracted_feature_list = experiment['feature_extraction_function'](llm_outputs_feature_extraction, selected_info_list)
            
            print(f"Number of mistaken extractions: {len(mistaken_extracted_idx)}")
            print(mistaken_extracted_idx)
            print(f"Number of extractions: {len(extracted_feature_list)}")
            print('+++++++++++++++++++++++++++++++++++++++++++++')
            for idx in  mistaken_conversation_idx:
                print(f"Mistaken conversation: {conversation_list[idx]}")
                print(f"Mistaken extraction: {extracted_feature_list[idx]}")
                print('--------------------------------')
            if len(mistaken_extracted_idx) == 0 and len(mistaken_conversation_idx) == 0:
                break

    print('saving outputs...')
    with open(experiment['output_filename_conversation'], 'w') as f:
        for output in conversation_list:
            json.dump(output, f)
            f.write('\n')
    with open(experiment['output_filename_feature_extraction'], 'w') as f:
        for output in llm_outputs_feature_extraction:
            json.dump(output, f)
            f.write('\n')
    print(f'outputs saved: \n {experiment["output_filename_conversation"]} \n {experiment["output_filename_feature_extraction"]}')
    print(len(mistaken_conversation_idx))
    assert len(mistaken_conversation_idx) == 0, f"There are still mistaken conversations: {["\n".join(conversation_list[i] for i in mistaken_conversation_idx)]}"
    assert len(mistaken_extracted_idx) == 0, f"There are still mistaken extractions: {["\n".join(extracted_feature_list[i] for i in mistaken_extracted_idx)]}"

    print('---------------------------------------- Dataset Generation Completed ----------------------------------------')
    # Cleanup
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    print("Process group destroyed.")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="First Experiment:")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="LLM Name.")
    parser.add_argument("--track", type=str, default="A", help="Track name: A:Arithmetic, S:Semantic, or T:Temporal.")
    parser.add_argument("--conv_type", type=str, default="Uni", help="Conversation type: Uni, or Multi.")
    parser.add_argument("--datasets_helping_folder", type=str, default="./Dataset_Generation/Dataset_Helping", help="Path to datasets folder of helping files (structured dataset).")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
 

    print("----- Experiment Setup ----")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"Number of GPUs: {num_gpus}")
    print('---------------------------')

    # Set up Hugging Face authentication
    login(token=os.environ.get("HF_TOKEN"))
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token")

    config = {
        **vars(args),
        "num_gpus": num_gpus,
    }
    main(**config)