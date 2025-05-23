# colbert_retriever.py
# A ColBERT-based retriever with BM25-compatible API

import os
import sys
import tarfile
import requests
from typing import List, Tuple

try:
    from colbert import Indexer, Searcher
except ImportError:
    os.system("pip install git+https://github.com/stanford-futuredata/ColBERT.git")
    try:
        from colbert import Indexer, Searcher
    except ImportError:
        print("ColBERT package not found. Please install it with:\n"
              "  pip install git+https://github.com/stanford-futuredata/ColBERT.git")
        sys.exit(1)

import torch
import numpy as np
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score

class ColBERTRetriever:

    def __init__(
        self,
        corpus: List[str],
        k: int = 5,
        checkpoint: str = 'colbert-ir/colbertv2.0'
    ):
        """
        Args:
            corpus: list of document strings (one passage per string).
            k: number of top documents to retrieve.
            checkpoint: ColBERT checkpoint name or path.
        """
        # 1. Build ColBERT config and load checkpoint
        self.checkpoint = checkpoint
        self.config = ColBERTConfig(doc_maxlen=510, nbits=2)
        self.ckpt = Checkpoint(self.checkpoint, colbert_config=self.config)


        # 5. Store corpus and k
        self.corpus = corpus
        self.k = k
        self.docs = self.ckpt.docFromText(self.corpus, bsize=32)[0]
        self.docs_mask = torch.ones(self.docs.shape[:2], dtype=torch.long)

    def retrieve_data(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Returns:
          - top_k_docs: list of the top-k 
          - scores: list of the top-k similarity scores
        """
        query =  self.ckpt.queryFromText([query])

        scores = colbert_score(query, self.docs, self.docs_mask).flatten().cpu().numpy().tolist()
        ranking = np.argsort(scores)[::-1]
        
        # Get top-k documents and their scores
        top_k_indices = ranking[:self.k]
        top_k_docs = [self.corpus[i] for i in top_k_indices]
        top_k_scores = [scores[i] for i in top_k_indices]
        
        return top_k_docs, top_k_scores

# --------------------------------------------------------------
#
#                           For Testing
#
# --------------------------------------------------------------
# if __name__ == "__main__":
#     # Example test
#     sample_corpus = [
#         "This is the first passage about information retrieval.",
#          "Finally, the third text mentions late interaction models.",
#         "Here's a second document describing ColBERT and semantic search."
#     ]
#     retriever = ColBERTRetriever(sample_corpus, k=2)
#     query = "what is late interaction?"
#     top_k_docs, top_k_scores = retriever.retrieve_data(query)
#     print(f"Query: {query!r}")
#     print("Top-k Documents:", top_k_docs)
#     print("Scores:", top_k_scores)
