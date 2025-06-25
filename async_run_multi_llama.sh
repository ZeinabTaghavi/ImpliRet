
export VLLM_WORKER_MULTIPROC_METHOD=fork

echo "Launch local vLLM server" 
# ------------------------------------------------------------------
# Start vLLM server via helper script (background) and wait for load
# ------------------------------------------------------------------
# run_tests.sh  (top of file)
PROJECT_ROOT=./RAG_Style   # adjust once
source "$PROJECT_ROOT/scripts/async/start_vllm.sh"

echo "Starting the job" 
python ./RAG_Style/scripts/async/async_run_tests.py \
       --config ./RAG_Style/experiment_configs/oracle_retriever/A_Multi_llama_1.yaml


# ------------------------------------------------------------------
# Shut down the vLLM server
# ------------------------------------------------------------------
echo "Stopping vLLM server (PID=$VLLM_PID)"
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

