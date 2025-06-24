import torch
from transformers import AutoTokenizer, AutoModel



class DragonPlusRetriever:

    def __init__(self, corpus, k=5):

        self.corpus = corpus
        self.k = k
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != 'cuda':
            print(f"Not Cuda, Using device: {self.device}")
        
        # 1. Loading the tokenizer and encoders
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        self.query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to(self.device)
        self.context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to(self.device)

        # Get max token length from corpus
        max_pos = self.context_encoder.config.max_position_embeddings
        self.max_token_length = min(int(max(len(self.tokenizer.encode(doc)) for doc in corpus) * 1.2), max_pos)
        # 2. Tokenizing the corpus
        tokenized_corpus = self.tokenizer(corpus, padding=True, truncation=True, max_length=self.max_token_length, return_tensors='pt')
        # Move tokenized corpus to device
        tokenized_corpus = {k: v.to(self.device) for k, v in tokenized_corpus.items()}

        # 3. Computing the embeddings
        with torch.no_grad():
            self.context_embeddings = self.context_encoder(**tokenized_corpus).last_hidden_state[:, 0, :]

    def retrieve_data(self, query):

        # 1. Tokenizing the query
        query_tokens = self.tokenizer(query, return_tensors='pt')
        # Move query tokens to device
        query_tokens = {k: v.to(self.device) for k, v in query_tokens.items()}
        
        # 2. Computing the embeddings
        with torch.no_grad():
            query_emb = self.query_encoder(**query_tokens).last_hidden_state[:, 0, :]
            
        # 3. Computing the similarity scores
        scores = query_emb @ self.context_embeddings.T
        
        # 4. Retrieving the top documents
        top_k_indices = torch.argsort(scores, dim=-1, descending=True)[:, :self.k]
        top_docs = [self.corpus[i] for i in top_k_indices[0]]
        top_scores = scores[0][top_k_indices[0]]

        return top_docs, top_scores