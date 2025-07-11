
export HF_HOME=...
export HF_TOKEN= ...


echo "----------------------------  BM25  ----------------------------------"
echo "Retrieving for track A, type Multi, retriever_name bm25"
python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name bm25


# echo "Retrieving for track A, type Uni, retriever_name bm25"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name bm25


# echo "----------------------------  ColBERT  ----------------------------------"
# echo "Retrieving for track A, type Multi, retriever_name colbert"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name colbert


# echo "Retrieving for track A, type Uni, retriever_name colbert"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name colbert


# echo "----------------------------  Contriever  ----------------------------------"
# echo "Retrieving for track A, type Multi, retriever_name contriever"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name contriever


# echo "Retrieving for track A, type Uni, retriever_name contriever"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name contriever


# echo "----------------------------  DragonPlus  ----------------------------------"
# echo "Retrieving for track A, type Multi, retriever_name dragonplus"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name dragonplus


# echo "Retrieving for track A, type Uni, retriever_name dragonplus"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name dragonplus



# echo "----------------------------  ReasonIR  ----------------------------------"
# echo "Retrieving for track A, type Uni, retriever_name reasonir"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name reasonir


# echo "Retrieving for track A, type Multi, retriever_name reasonir"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name reasonir


# echo "----------------------------  Hippo  ----------------------------------"
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# pip install hipporag
# vllm serve meta-llama/Llama-3.3-70B-Instruct \
#     --tensor-parallel-size 2 \
#     --max_model_len 4096 \
#     --gpu-memory-utilization 0.95 \
#     --api-key "" &
# # ------------------------------------------------------------------
# # Wait for vLLM /health endpoint to return HTTP 200 (engine ready)
# # ------------------------------------------------------------------
# echo "[start_vllm] Waiting for vLLM /health…"
# until curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health" | grep -q "^200$"; do
#   sleep 1
#   echo "[start_vllm] Waiting for vLLM /health…"
# done
# echo "[start_vllm] vLLM is ready!"

# echo "Retrieving for track A, type Multi, retriever_name hipporag"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse multispeaker --retriever_name hipporag


# echo "Retrieving for track A, type Uni, retriever_name hipporag"
# python ./Retrieval/retrieve_indexing.py  --output_folder ./Retrieval/results/ --category arithmetic --discourse unispeaker --retriever_name hipporag



# # Reporting
# python Retrieval/reporting.py