#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# [중요] 수정된 Python 스크립트 경로
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLLM_inference-vllm-single.py"

# 데이터 및 스키마 경로
INPUT_PATH="/data/minseo/experiments4/data/dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_singleturn.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy.json"

# 결과 저장 루트 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_output"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_logs"

# 파일명 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test_vllm_1"

# GPU 및 서버 설정
GPU_ID=0,1,2,3
PORT=8001
VLLM_URL="http://localhost:$PORT/v1"
CONCURRENCY=50  # 비동기 요청 동시 처리 수

# ==============================================================================
# 2. 실험 변수 (모델 및 프롬프트 설정)
# ==============================================================================

MODELS=(
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
    # # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
)

# 실험 조건 반복 리스트
PROMPT_TYPES=("imp-zs" ) #"imp-pref-group"
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=4 # Tensor Parallelism Size (GPU 1장 = 1)

# ==============================================================================
# 3. 헬퍼 함수
# ==============================================================================

# 스크립트 강제 종료(Ctrl+C) 시 서버도 같이 죽이기 위한 Trap 설정
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
    MAX_RETRIES=60 # 5분 대기 (5s * 60)
    COUNT=0
    
    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        # 서버 프로세스가 죽었는지 확인
        if ! ps -p $SERVER_PID > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server.log
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
# 4. 메인 루프 실행
# ==============================================================================

echo "========================================================"
echo "Automated Batch Inference Started at $(date)"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    # 모델명에서 슬래시(/)를 언더바(_)로 치환하여 폴더명으로 사용
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # ---------------------------------------------------------
    # 4-1. 모델별 Tool Parser 설정 (필요시 활성화)
    # ---------------------------------------------------------
    PARSER_FLAG=""
    
    if [[ "$model" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    elif [[ "$model" == *"Qwen"* ]] || [[ "$model" == *"Hermes"* ]]; then
        PARSER_FLAG="--tool-call-parser hermes" 
    else
        PARSER_FLAG="--tool-call-parser hermes" 
        #PARSER_FLAG="" 
    fi

    # ---------------------------------------------------------
    # 4-2. vLLM 서버 백그라운드 실행
    # ---------------------------------------------------------
    # --gpu-memory-utilization 0.9: OOM 방지용 여유분
    # --max-model-len: 컨텍스트 길이 (모델 스펙에 맞게 조절)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code > vllm_server.log 2>&1 &
    
    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"
    echo ">> Logs are being written to vllm_server.log"

    # ---------------------------------------------------------
    # 4-3. 서버 준비 대기
    # ---------------------------------------------------------
    wait_for_server

    # ---------------------------------------------------------
    # 4-4. Python Client 실행 (루프)
    # ---------------------------------------------------------
    echo "[STEP 2] Running Python Client Scripts..."
    
    for context in "${CONTEXT_TYPES[@]}"; do
        for prompt_type in "${PROMPT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo " >> [Processing] Context: $context | Prompt: $prompt_type | Pref: $pref"

                # 결과 저장 경로 생성
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE_NAME/$prompt_type"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_${TEST_TAG}.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_${TEST_TAG}.log"

                # Python 스크립트 실행
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --query_path "$QUERY_PATH" \
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
    # 4-5. 서버 종료 및 정리
    # ---------------------------------------------------------
    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    SERVER_PID="" # PID 초기화
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    
    # 다음 모델 로딩 전 잠시 대기 (GPU 메모리 해제 보장)
    sleep 10

done

echo "========================================================"
echo "All Jobs Finished Successfully at $(date)"
echo "========================================================"