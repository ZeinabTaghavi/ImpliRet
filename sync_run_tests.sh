
# export HF_HOME=...
# export HF_TOKEN= ...
export VLLM_WORKER_MULTIPROC_METHOD=fork

echo "Starting the job" 

python ./RAG_Style/scripts/syncr/sync_run_tests.py \
       --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_3b_1.yaml

# python ./RAG_Style/scripts/syncr/sync_run_tests.py \
#        --config ./RAG_Style/experiment_configs/bm/A_Multi_llama_bm_1.yaml

# python ./RAG_Style/scripts/syncr/sync_run_tests.py \
#        --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_1.yaml

