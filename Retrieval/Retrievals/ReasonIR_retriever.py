import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from accelerate import infer_auto_device_map, dispatch_model

class ReasonIRRetriever:
    def __init__(
        self,
        corpus: list[str],
        k: int = 5,
        llm_model_name: str = "reasonir/ReasonIR-8B",
    ):
        self.corpus = corpus
        self.k = k
        self.llm_model_name = llm_model_name

        # 1) Tell Accelerate to shard & quantize the model across GPUs/CPU
        self.model_kwargs = {
            "device_map": "auto",          # shard layers to all available GPUs (then CPU/disk)  [oai_citation:10‡huggingface.co](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference?utm_source=chatgpt.com)
            "torch_dtype": torch.bfloat16, # use BF16 for FP performance
            "load_in_8bit": True,          # quantize weights to 8-bit (requires bitsandbytes)  [oai_citation:11‡medium.com](https://medium.com/%40syedhamzatahir1001/how-hugging-faces-accelerate-helps-us-handle-huge-models-97ae9fe32fa6?utm_source=chatgpt.com)
            "low_cpu_mem_usage": True,     # reduce peak RAM during load
        }
        # 2) Load & dispatch the SentenceTransformer
        self.model = SentenceTransformer(
            self.llm_model_name,
            trust_remote_code=True,
            device="cuda",           # entry point for Accelerate dispatch
            model_kwargs=self.model_kwargs,
        )
        # 3) (Optional) enforce pooling to ignore prompt in embeddings
        self.model.set_pooling_include_prompt(include_prompt=False)

        # 4) Build the corpus index with batched, off-loaded embeddings
        self._build_doc_index(batch_size=32)

    def _build_doc_index(self, batch_size: int = 32):
        """
        Encode the corpus in mini-batches, move embeddings to CPU immediately,
        and clear GPU cache to prevent OOM.
        """
        self.doc_embeddings = []
        for start in range(0, len(self.corpus), batch_size):
            batch_docs = self.corpus[start : start + batch_size]
            with torch.no_grad():
                embeds = self.model.encode(
                    batch_docs,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.model.device,  # Accelerate will place each chunk correctly
                )
            # Move to CPU and free GPU memory
            self.doc_embeddings.append(embeds.cpu())
            del embeds
            torch.cuda.empty_cache()
        # Stack into one big CPU matrix
        self.doc_embeddings = torch.vstack(self.doc_embeddings)

    def retrieve_data(self, query: str):
        """
        Encode the query (automatically placed on the right GPU),
        then compute dot-product scores on CPU.
        Returns top-k docs and their scores.
        """
        # 1) Encode + off-load
        query_emb = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.model.device,
        ).cpu()

        # 2) CPU matmul avoids any cross-GPU errors
        scores = (query_emb @ self.doc_embeddings.T).tolist()

        # 3) Select top-k
        top_idx = np.argsort(scores)[::-1][: self.k]
        top_docs = [self.corpus[i] for i in top_idx]
        top_scores = [scores[i] for i in top_idx]

        return top_docs, top_scores