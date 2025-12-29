#!/bin/bash

# ==============================================================================
# 1. Environment & Paths Configuration
# ==============================================================================

# [IMPORTANT] Path to the Multi-turn vLLM Python script created earlier
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLLM_inference-vllm-multi.py"

# Data & Schema Paths
INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"

# [Updated] Using Multi-turn Query Path
MULTITURN_PATH="/data/minseo/experiments4/query_multiturn-domain.json"

PREF_LIST_PATH="/data/minseo/experiments4/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_all.json"

# Output & Log Base Directories
BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/1230-1_output"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/1230-1_logs"

# Tag Settings
DATE_TAG="$(date +%m%d)"
TEST_TAG="test_vllm_1"

# GPU & Server Settings
GPU_ID=0,1,2,3
PORT=8002          # Changed port to avoid conflict with single-turn script
VLLM_URL="http://localhost:$PORT/v1"
CONCURRENCY=50     # Async concurrency limit

# ==============================================================================
# 2. Experiment Variables (Models & Prompts)
# ==============================================================================

MODELS=(
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    # # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
)

# Experiment Loops
PROMPT_TYPES=("imp-zs") # Options: "imp-zs", "imp-fs", "imp-pref-group"
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=4 # Tensor Parallelism Size (4 GPUs)

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

# Trap signals to ensure server is killed on exit
trap cleanup SIGINT SIGTERM ERR

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ""
        echo "[WARN] Killing vLLM Server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    exit 1
}

wait_for_server() {
    echo "Waiting for vLLM server at $VLLM_URL..."
    MAX_RETRIES=60 # Wait up to 5 minutes
    COUNT=0
    
    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        # Check if server process died
        if ! ps -p $SERVER_PID > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server_multi.log
            exit 1
        fi

        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            kill $SERVER_PID
            exit 1
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. Main Execution Loop
# ==============================================================================

echo "========================================================"
echo "Automated Batch Inference (Multi-turn) Started at $(date)"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    # Safe model name for directory structure
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # ---------------------------------------------------------
    # 4-1. Model-specific Tool Parser Flag
    # ---------------------------------------------------------
    PARSER_FLAG=""
    
    if [[ "$model" == *"Llama"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    elif [[ "$model" == *"Qwen"* ]] || [[ "$model" == *"Hermes"* ]]; then
        PARSER_FLAG="--tool-call-parser hermes" 
    else
        PARSER_FLAG="--tool-call-parser hermes" 
    fi

    # ---------------------------------------------------------
    # 4-2. Start vLLM Server in Background
    # ---------------------------------------------------------
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code > vllm_server_multi.log 2>&1 &
    
    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"
    echo ">> Logs are being written to vllm_server_multi.log"

    # ---------------------------------------------------------
    # 4-3. Wait for Server Ready
    # ---------------------------------------------------------
    wait_for_server

    # ---------------------------------------------------------
    # 4-4. Run Python Client (Loop through conditions)
    # ---------------------------------------------------------
    echo "[STEP 2] Running Python Client Scripts..."
    
    for context in "${CONTEXT_TYPES[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo " >> [Processing] Context: $context | Prompt: $prompt_type | Pref: $pref"

                # Generate Output Directories (multi-turn path structure)
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME/$prompt_type"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/multiturn-query/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.log"

                # Run Python Script with --multiturn_path
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --multiturn_path "$MULTITURN_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --vllm_url "$VLLM_URL" \
                    --concurrency "$CONCURRENCY"

            done
        done
    done

    # ---------------------------------------------------------
    # 4-5. Stop Server & Cleanup
    # ---------------------------------------------------------
    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    SERVER_PID="" # Reset PID
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    
    # Wait for GPU memory release
    sleep 10

done

echo "========================================================"
echo "All Jobs Finished Successfully at $(date)"
echo "========================================================"