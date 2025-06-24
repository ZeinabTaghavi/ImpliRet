import os
import json
import argparse
import shutil
from tqdm import tqdm
from datasets import load_dataset

def save_retriever_indices(output_folder, category, discourse, retriever_name):

    
    hf_dataset = load_dataset("zeinabTaghavi/ImpliRet", name=discourse, split=category)

    dataset = [item for item in hf_dataset]  # convert to list of dicts

    
    user_corpus= []
    user_questions = []
    user_gold_indices = []
    for idx, item in enumerate(dataset):
        qs = item["question"]
        pos_document = item["pos_document"]

        user_corpus.append(pos_document)
        user_questions.append(qs)
        user_gold_indices.append(idx)
   

    if retriever_name.lower() == "bm25":
        try:
            from retrievals.BM_retriever import BM25Retriever
            retriever_module = BM25Retriever
        except:
            raise Exception("BM25Retriever not found")
    elif retriever_name.lower() == "dragonplus":
        try:
            from retrievals.DragonPlus_retriever import DragonPlusRetriever
            retriever_module = DragonPlusRetriever
        except:
            raise Exception("DragonPlusRetriever not found")
    elif retriever_name.lower() == "contriever":
        try:
            from retrievals.Contriever_retriever import ContrieverRetriever
            retriever_module = ContrieverRetriever
        except:
            raise Exception("ContrieverRetriever not found")
    elif retriever_name.lower() == "colbert":
        try:
            from retrievals.ColBERT_retriever import ColBERTRetriever
            retriever_module = ColBERTRetriever
        except:
            raise Exception("ColBERTRetriever not found")
    elif retriever_name.lower() == "reasonir":
        try:
            from retrievals.ReasonIR_retriever import ReasonIRRetriever
            retriever_module = ReasonIRRetriever
        except:
            raise Exception("ReasonIRRetriever not found")
    elif retriever_name.lower() == "hipporag":
        try:
            from retrievals.HippoRag2_retriever import HippoRAG2Retriever
            retriever_module = HippoRAG2Retriever
        except:
            raise Exception("HippoRAG2Retriever not found")
    else:
        raise Exception("Invalid retriever name")

    results = []


    retriever = retriever_module(user_corpus, k=len(user_corpus))
    
    for user_id in tqdm(range(len(user_corpus)), desc="Processing questions"):
        top_docs, scores = retriever.retrieve_data(user_questions[user_id])
        if len(top_docs) == 0:
            raise Exception("No top docs")
        else:
            if len(scores) == 1:
                scores = scores[0]
                assert len(scores) == len(top_docs)
            retrieved_indices = [[user_corpus.index(doc), round(float(score), 4)] for doc, score in zip(top_docs, scores)]

        assert len(list(set(top_docs))) == len(user_corpus)
        assert len(list(set([items[0] for items in retrieved_indices]))) == len(user_corpus)

        results.append({
            "question": user_questions[user_id],
            "gold_index": user_gold_indices[user_id],
            "index_score_tuple_list": retrieved_indices
        })
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{category}_{discourse}_{retriever_name}_index.jsonl")
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="./Retrieval/Results/")
    parser.add_argument("--category", type=str, choices=["temporal", "arithmetic", "wknowledge"], default="arithmetic")
    parser.add_argument("--discourse", type=str, choices=["unispeaker", "multispeaker"], default="unispeaker")
    parser.add_argument("--retriever_name", type=str, default="bm25", help="bm25 or dragonplus or contriever or reasonir or hipporag")
    args = parser.parse_args()

    save_retriever_indices(args.output_folder, args.category, args.discourse, args.retriever_name)


