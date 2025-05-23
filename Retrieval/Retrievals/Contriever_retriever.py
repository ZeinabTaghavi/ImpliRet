import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class ContrieverRetriever:
    def __init__(
        self,
        corpus: List[str],
        model_name: str = "facebook/contriever",
        k: int = 5,
        device: str = None
    ):
        """
        Args:
            corpus: list of document strings.
            model_name: HF model identifier (e.g. "facebook/contriever").
            k: number of top documents to retrieve.
            device: torch device (defaults to CUDA if available).
        """
        # 1. Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 2. Embed the corpus once
        inputs = self.tokenizer(
            corpus,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, dim)
        attention_mask = inputs["attention_mask"]      # (batch_size, seq_len)
        self.corpus_embeddings = self._mean_pooling(token_embeddings, attention_mask)
        self.corpus = corpus
        self.k = k

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean‐pooling over token embeddings, taking attention mask into account.
        """
        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Zero out embeddings where mask=0
        token_embeddings = token_embeddings * mask
        summed = token_embeddings.sum(dim=1)             # (batch_size, dim)
        counted = mask.sum(dim=1).clamp(min=1e-9)        # avoid div by zero
        return summed / counted                          # (batch_size, dim)

    def retrieve_data(
        self,
        query: str
    ) -> Tuple[List[str], List[float]]:
        """
        Returns the top‐k documents and their cosine‐similarity scores for a given query.
        """
        # 1. Embed the query
        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_emb = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

        # 2. Compute cosine similarities
        # corpus_embeddings: (N, dim), query_emb: (1, dim)
        scores = torch.nn.functional.cosine_similarity(
            query_emb, self.corpus_embeddings, dim=-1
        )  # (N,)

        # 3. Select top‐k
        topk = torch.topk(scores, k=min(self.k, len(self.corpus)), largest=True)
        indices = topk.indices.cpu().tolist()
        top_scores = topk.values.cpu().tolist()
        top_docs = [self.corpus[i] for i in indices]
        return top_docs, top_scores