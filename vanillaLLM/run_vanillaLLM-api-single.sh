#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
# [중요] 실행할 파이썬 파일명 확인
PYTHON_SCRIPT="/data/minseo/experiments4/vanillaLLM/vanillaLLM_inference-api-single.py"

# 데이터 및 스키마 경로
INPUT_PATH="/data/minseo/experiments4/data/dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_singleturn.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_easy.json"

# 출력 및 로그 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_output"
BASE_LOG_DIR="/data/minseo/experiments4/vanillaLLM/inference/1229-3_logs"

# 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test_1"

# 동시성 설정
CONCURRENCY=5

# ==============================================================================
# 2. 실험 변수 (모델 및 프롬프트)
# ==============================================================================

# [Model List]
MODELS=(
    #"gpt-4o-mini-2024-07-18"  # 일반 모델
    "gemini-3-flash-preview"          
    "gpt-5-mini"              
    #"gpt-5.1"                 
)

# [Prompt Types]
PROMPT_TYPES=("imp-zs") #"imp-pref-group"

# [Context Types]
CONTEXT_TYPES=("diag-apilist")

# [Preference Types]
PREF_TYPES=("easy" "medium" "hard") #"low" "medium"

# ==============================================================================
# 3. 배치 실행 로직
# ==============================================================================

echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "Models: ${MODELS[*]}"
echo "Prompts: ${PROMPT_TYPES[*]}"
echo "========================================================"

for model in "${MODELS[@]}"; do
    # 모델명 안전 변환 (슬래시를 언더스코어로)
    MODEL_SAFE_NAME="${model//\//_}"

    # ------------------------------------------------------------------
    # [Reasoning Effort 설정 로직]
    # 모델명에 gpt-5, gemini-3, o1, o3 등이 포함되면 Reasoning 레벨 루프를 돔
    # 그 외 일반 모델은 'default' (설정 없음)로 1회 실행
    # ------------------------------------------------------------------
    EFFORT_LEVELS=("default")  # 기본값 (설정 없음)
    if [[ "$model" == *"gpt-5"* ]] || [[ "$model" == *"gemini-3"* ]] || [[ "$model" == *"o1"* ]] || [[ "$model" == *"o3"* ]]; then
        EFFORT_LEVELS=("minimal") #"low" "medium" "high"
    fi

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                for effort in "${EFFORT_LEVELS[@]}"; do

                    echo ""
                    echo "--------------------------------------------------------------------------------"
                    echo "[RUNNING] Model: $model | Prompt: $prompt_type | Pref: $pref | Effort: $effort"
                    echo "--------------------------------------------------------------------------------"

                    # ------------------------------------------------------------------
                    # 디렉토리 및 파일명 생성 (Effort 폴더 추가)
                    # ------------------------------------------------------------------
                    # 구조: BASE / context / pref / model / prompt / effort / filename
                    OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type/$effort"
                    LOG_DIR="$BASE_LOG_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME/$prompt_type/$effort"
                    
                    mkdir -p "$OUTPUT_DIR"
                    mkdir -p "$LOG_DIR"

                    FILENAME="${DATE_TAG}_${TEST_TAG}.json"
                    LOGNAME="${DATE_TAG}_${TEST_TAG}.jsonl"

                    OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                    LOG_FILE="$LOG_DIR/$LOGNAME"

                    # ------------------------------------------------------------------
                    # Python 명령어 구성 (effort 인자 처리)
                    # ------------------------------------------------------------------
                    CMD="python $PYTHON_SCRIPT \
                        --input_path $INPUT_PATH \
                        --query_path $QUERY_PATH \
                        --pref_list_path $PREF_LIST_PATH \
                        --pref_group_path $PREF_GROUP_PATH \
                        --tools_schema_path $TOOLS_SCHEMA_PATH \
                        --context_type $context \
                        --pref_type $pref \
                        --prompt_type $prompt_type \
                        --model_name $model \
                        --output_path $OUTPUT_FILE \
                        --log_path $LOG_FILE \
                        --concurrency $CONCURRENCY"

                    # default가 아니면 reasoning_effort 인자 추가
                    if [ "$effort" != "default" ]; then
                        CMD="$CMD --reasoning_effort $effort"
                    fi

                    # ------------------------------------------------------------------
                    # 실행
                    # ------------------------------------------------------------------
                    eval $CMD

                    # 실행 결과 확인
                    if [ $? -eq 0 ]; then
                        echo ">> [SUCCESS] Saved to: $OUTPUT_FILE"
                    else
                        echo ">> [ERROR] Failed at Model: $model (Effort: $effort)"
                    fi
                done
            done
        done
    done
done

echo ""
echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"