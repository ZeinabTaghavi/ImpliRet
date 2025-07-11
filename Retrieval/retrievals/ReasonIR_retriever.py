

from transformers import AutoModel, AutoTokenizer


import numpy as np
import torch

class ReasonIRRetriever:

    def __init__(self, 
                corpus: list[str], 
                k: int=5, 
                llm_model_name: str = "reasonir/ReasonIR-8B"):

        self.corpus = corpus
        self.k = k
        self.llm_model_name = llm_model_name
        
        # Get the device of the model
        self.model = AutoModel.from_pretrained(self.llm_model_name, torch_dtype="auto", trust_remote_code=True)
        self.model.to("cuda")
        self.model.eval()


        self.doc_instruction = ""
        self.doc_embeddings = [self.model.encode(doc, instruction=self.doc_instruction) for doc in self.corpus]


    def retrieve_data(self, query: str):
        """
        Retrieve top-k documents for a single query string.
        Returns:
            top_docs (list[str]): the retrieved document texts.
            top_scores (list[float]): the retrieval scores.
        """
        # perform retrieval for one query
        query_instruction = ""
        query_emb = self.model.encode(query, instruction=query_instruction)

        scores = [query_emb @ doc_embedding.T for doc_embedding in self.doc_embeddings]
        top_docs = [self.corpus[i] for i in np.argsort(scores)[::-1][:self.k]]
        top_scores = [scores[i] for i in np.argsort(scores)[::-1][:self.k]]
        return top_docs, top_scores
