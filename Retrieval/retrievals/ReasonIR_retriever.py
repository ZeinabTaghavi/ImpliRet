
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence_transformers is not installed. Please install it using 'pip install sentence-transformers'.")
    exit(1)

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
        self.model_kwargs = {"torch_dtype": "auto", 'device_map': "auto"}
        
        # Get the device of the model
        self.model = SentenceTransformer(self.llm_model_name, trust_remote_code=True, model_kwargs=self.model_kwargs)
        self.device = next(self.model.parameters()).device

        self.model.set_pooling_include_prompt(include_prompt=False) # exclude the prompt during pooling

        doc_instruction = ""
        self.doc_embeddings = [self.model.encode(doc, instruction=doc_instruction) for doc in self.corpus]
        # Move all embeddings to the same device as the model
        self.doc_embeddings = [torch.tensor(emb, device=self.device) for emb in self.doc_embeddings]

    def retrieve_data(self, query: str):
        """
        Retrieve top-k documents for a single query string.
        Returns:
            top_docs (list[str]): the retrieved document texts.
            top_scores (list[float]): the retrieval scores.
        """
        # perform retrieval for one query
        query_instruction = ""
        query_embedding = self.model.encode(query, instruction=query_instruction)
        # Move query embedding to the same device as the model
        query_embedding = torch.tensor(query_embedding, device=self.device)
        scores = [query_embedding @ doc_embedding.T for doc_embedding in self.doc_embeddings]
        top_docs = [self.corpus[i] for i in np.argsort(scores)[::-1][:self.k]]
        top_scores = [scores[i] for i in np.argsort(scores)[::-1][:self.k]]
        return top_docs, top_scores

        