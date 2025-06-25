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

from multiprocessing import Pool
import bm25s


class BM25Retriever:

    def __init__(self, corpus, k=5):

        # 1. Tokenizing the corpus
        # Assert that all items in corpus are strings
        assert all(isinstance(item, str) for item in corpus), "All items in corpus must be strings"
        tokenized_corpus = bm25s.tokenize(corpus)
        # 2. Indexing the corpus
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(tokenized_corpus)
        # 3. Setting the number of top documents to retrieve
        self.k = k

    def retrieve_data(self, query):

        retrive_k = self.k

        for i in range(self.k):
            try:
                # 1. Tokenizing the query
                query_tokens = bm25s.tokenize(query)
                # 2. Retrieving the top documents
                top_docs, scores = self.retriever.retrieve(query_tokens, k=retrive_k)
                return top_docs[0], scores
            except:
                if retrive_k > 1:
                    retrive_k = retrive_k - 1
                    print(f"Retrieving {retrive_k} documents")
                    continue
                else:
                    return [], []

