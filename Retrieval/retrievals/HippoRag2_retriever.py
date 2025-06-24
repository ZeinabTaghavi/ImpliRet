'''
- This script is used to implement the BM25 algorithm to retrieve the most relevant documents from the corpus.
- This script can be called from the controller.py while interpreting the LLM's functionacall script to store the
corpus and retrieve the most relevant documents from the corpus.
- The structire of the controller given input is as follows:
STORE:
Conversation
QUERY:
What task is Aquila scheduled to do on 2020-05-01 at 17 o'clock?
- Here we interpret the input and store the data in the Temporary Knowledge Base.
'''

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
import os
import shutil

class HippoRAG2Retriever:

    def __init__(self, 
                corpus: list[str], 
                k: int=5, 
                save_dir: str ='Retrieval/retrievals/HippoRAG_outputs',
                llm_model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
                embedding_model_name: str = 'facebook/contriever'):
        

         # Clean the cache of hipporag, it keeps the docs from previous runs
        
        if os.path.exists('Retrieval/retrievals/HippoRAG_outputs'):
            shutil.rmtree('Retrieval/retrievals/HippoRAG_outputs')
            os.makedirs('Retrieval/retrievals/HippoRAG_outputs')
        # Clear CUDA cache to free up GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        self.corpus = corpus
        self.k = k
        self.save_dir = save_dir
        self.llm_base_url = "http://localhost:8000/v1"
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

        
        # configure OpenIE saving
        cfg = BaseConfig(save_openie=True)
        self.hipporag = HippoRAG(
            global_config=cfg,
            save_dir=self.save_dir,
            llm_model_name=self.llm_model_name,
            llm_base_url=self.llm_base_url,
            embedding_model_name=self.embedding_model_name,
        )

        # Run indexing
        self.hipporag.index(docs=self.corpus)

    def retrieve_data(self, query: str):
        """
        Retrieve top-k documents for a single query string.
        Returns:
            top_docs (list[str]): the retrieved document texts.
            top_scores (list[float]): the retrieval scores.
        """
        # perform retrieval for one query
        solutions = self.hipporag.retrieve(queries=[query], num_to_retrieve=self.k)
        if not solutions:
            return [], []
        sol = solutions[0]
        # convert scores to Python list
        scores = sol.doc_scores.tolist() if hasattr(sol.doc_scores, "tolist") else list(sol.doc_scores)
        top_docs = []
        for i in sol.docs:
            top_docs.append(i)
        return top_docs, scores

        