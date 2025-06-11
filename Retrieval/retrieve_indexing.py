import os
import json
import argparse
import shutil


def save_retriever_indices(dataset_folder, output_folder, track, conv_type, retriever_name, question_type, expanded_question_folder):
    dataset_path = os.path.join(dataset_folder, f"{track}_{conv_type}.jsonl")
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    if question_type.lower() == "expanded":
        with open(os.path.join(expanded_question_folder, f"{track}_{conv_type}.jsonl"), "r") as f:
            expanded_questions = [json.loads(line) for line in f]


    user_corpus_dict = {}
    user_questions = {}
    user_gold_indices = {}
    for idx, item in enumerate(dataset):
        user_id = item["user_ID"]
        context = item["context"]
        if question_type.lower() == "expanded":
            qs = expanded_questions[idx]["rewritten_question"]
            assert expanded_questions[idx]['original_question'] == item["question"]
        else:
            qs = item["question"]

        if user_id not in user_corpus_dict:
            user_corpus_dict[user_id] = []
            user_questions[user_id] = []
            user_gold_indices[user_id] = []
   
        user_gold_indices[user_id].append(len(user_corpus_dict[user_id]))
        user_corpus_dict[user_id].append(context)
        user_questions[user_id].append(qs)




    if retriever_name.lower() == "bm25":
        try:
            from Retrievals.BM_retriever import BM25Retriever
            retriever_module = BM25Retriever
        except:
            raise Exception("BM25Retriever not found")
    elif retriever_name.lower() == "dragonplus":
        try:
            from Retrievals.DragonPlus_retriever import DragonPlusRetriever
            retriever_module = DragonPlusRetriever
        except:
            raise Exception("DragonPlusRetriever not found")
    elif retriever_name.lower() == "contriever":
        try:
            from Retrievals.Contriever_retriever import ContrieverRetriever
            retriever_module = ContrieverRetriever
        except:
            raise Exception("ContrieverRetriever not found")
    elif retriever_name.lower() == "colbert":
        try:
            from Retrievals.ColBERT_retriever import ColBERTRetriever
            retriever_module = ColBERTRetriever
        except:
            raise Exception("ColBERTRetriever not found")
    elif retriever_name.lower() == "reasonir":
        try:
            from Retrievals.ReasonIR_retriever import ReasonIRRetriever
            retriever_module = ReasonIRRetriever
        except:
            raise Exception("ReasonIRRetriever not found")
    elif retriever_name.lower() == "hipporag":
        try:
            from Retrievals.HippoRag2_retriever import HippoRAG2Retriever
            retriever_module = HippoRAG2Retriever
        except:
            raise Exception("HippoRAG2Retriever not found")
    else:
        raise Exception("Invalid retriever name")

    results = []
    for user_id in list(user_corpus_dict.keys())[:10]:
        corpus = user_corpus_dict[user_id]
        # Clean the cache of hipporag, it keeps the docs from previous runs
        if retriever_name.lower() == "hipporag":
            if os.path.exists('MetatagIndexing/Experiments/Retrieval'):
                shutil.rmtree('MetatagIndexing/Experiments/Retrieval')
                os.makedirs('MetatagIndexing/Experiments/Retrieval')
            # Clear CUDA cache to free up GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        retriever = retriever_module(corpus, k=len(corpus))
        for q_idx, question in enumerate(user_questions[user_id]):
            top_docs, scores = retriever.retrieve_data(question)
            if len(top_docs) == 0:
                raise Exception("No top docs")
            else:
                if len(scores) == 1:
                    scores = scores[0]
                    assert len(scores) == len(top_docs)
                retrieved_indices = [[corpus.index(doc), round(float(score), 4)] for doc, score in zip(top_docs, scores)]

            assert len(list(set(top_docs))) == len(corpus)
            assert len(list(set([items[0] for items in retrieved_indices]))) == len(corpus)

            results.append({
                "user_ID": user_id,
                "question": question,
                "gold_index": user_gold_indices[user_id][q_idx],
                "index_score_tuple_list": retrieved_indices
            })

    os.makedirs(output_folder, exist_ok=True)
    if question_type.lower() == "expanded":
        output_path = os.path.join(output_folder, f"{track}_{conv_type}_{retriever_name}_expanded_questions_index.jsonl")
    else:   
        output_path = os.path.join(output_folder, f"{track}_{conv_type}_{retriever_name}_index.jsonl")
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"ResuTehran0021189196.lts saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="./Dataset_Generation/Data/")
    parser.add_argument("--output_folder", type=str, default="./MetatagIndexing/Experiments/Retrieval/Results/")
    parser.add_argument("--track", type=str, default="A")
    parser.add_argument("--type", type=str, default="Uni")
    parser.add_argument("--retriever_name", type=str, default="bm25", help="bm25 or dragonplus or contriever or reasonir")
    parser.add_argument("--question_type", type=str, default="original", help="original or expanded")
    parser.add_argument("--expanded_question_folder", type=str, default="./Retrieval/Bright_Questions", help="path to expanded question folder")
    args = parser.parse_args()

    save_retriever_indices(args.dataset_folder, args.output_folder, args.track, args.type, args.retriever_name, args.question_type, args.expanded_question_folder)
