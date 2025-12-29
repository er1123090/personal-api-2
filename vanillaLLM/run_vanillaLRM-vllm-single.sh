#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# [중요] 실행할 Python 스크립트 경로 (위에서 작성한 파이썬 파일명으로 변경하세요)
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLRM_inference-vllm-single.py"

# 데이터 및 스키마 경로 (사용자 환경에 맞게 유지)
INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_singleturn.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy.json"

# 출력 및 로그 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_output"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_logs"

# 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test_deepseek_integration"

# vLLM 서버 설정
GPU_ID=0,1,2,3
PORT=8001
VLLM_URL="http://localhost:$PORT/v1"
CONCURRENCY=50  # Python 스크립트의 비동기 요청 수

# ==============================================================================
# 2. 실험 변수 (모델 및 프롬프트 설정)
# ==============================================================================

# 테스트할 모델 목록
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
)

# 실험 조건
PROMPT_TYPES=("imp-zs" ) #"imp-pref-group"
CONTEXT_TYPES=("diag-apilist")
PREF_TYPES=("easy" "medium" "hard")

TP_SIZE=4

# ==============================================================================
# 3. 안전 장치 (스크립트 강제 종료 시 서버 킬)
# ==============================================================================
cleanup() {
    echo ""
    echo "[Cleanup] Stopping vLLM Server (PID: $SERVER_PID)..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    echo "[Cleanup] Done."
    exit
}
trap cleanup SIGINT SIGTERM

# ==============================================================================
# 4. 헬퍼 함수: 서버 대기
# ==============================================================================
wait_for_server() {
    echo "Waiting for vLLM server to start at $VLLM_URL..."
    local retries=0
    local max_retries=60  # 5분 대기 (5s * 60)

    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        retries=$((retries+1))
        if [ $retries -ge $max_retries ]; then
            echo ""
            echo "[Error] Server failed to start within timeout."
            cleanup
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 5. 메인 루프
# ==============================================================================
echo "========================================================"
echo "Automated Batch Inference Started at $(date)"
echo "GPU: $GPU_ID | Port: $PORT | Concurrency: $CONCURRENCY"
echo "========================================================"

for model in "${MODELS[@]}"; do
    # 모델명에서 슬래시(/)를 언더바(_)로 치환하여 파일명에 사용
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 5-1. 모델별 Tool Parser 설정
    # vLLM 버전에 따라 지원하는 파서가 다를 수 있습니다.
    if [[ "$model" == *"Llama"* ]]; then
        PARSER="llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER="mistral"
    elif [[ "$model" == *"Qwen"* ]]; then
        PARSER="hermes" 
    else
        # 기본값: 모델 템플릿에 의존하거나 hermes 사용
        PARSER="hermes"
    fi
    
    echo ">> Selected Tool Parser: $PARSER"

    # 5-2. vLLM 서버 백그라운드 실행
    # --enable-auto-tool-choice: Tool 사용 여부 자동 결정
    # --max-model-len: 메모리 부족 시 조절 (예: 8192, 4096)
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        --tool-call-parser $PARSER \
        --max-model-len 8192 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 > "vllm_server_${MODEL_SAFE_NAME}.log" 2>&1 &
        
    SERVER_PID=$!
    echo ">> Server PID: $SERVER_PID"

    # 5-3. 서버 준비 대기
    wait_for_server

    # 5-4. 추론 스크립트 실행 (Client)
    echo "[STEP 2] Running Python Client Scripts..."
    
    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo "   -----------------------------------------------------"
                echo "   >> Prompt: $prompt_type | Context: $context | Pref: $pref"

                # 출력 경로 생성
                OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type"
                LOG_DIR="$BASE_LOG_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type"
                mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

                FILENAME="${DATE_TAG}_${TEST_TAG}.json"
                LOGNAME="${DATE_TAG}_${TEST_TAG}.jsonl"
                
                OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                LOG_FILE="$LOG_DIR/$LOGNAME"

                # Python 실행
                # 클라이언트는 비동기 요청을 보내므로 별도 GPU 할당 필요 없음 (CPU 사용)
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --prompt_type "$prompt_type" \
                    --model_name "$model" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --vllm_url "$VLLM_URL" \
                    --concurrency "$CONCURRENCY"

            done
        done
    done

    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    SERVER_PID="" # PID 초기화
    echo ">> Server Stopped."
    echo ""
    
    # 포트 반환 및 정리를 위한 대기 시간
    sleep 10

done

echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"