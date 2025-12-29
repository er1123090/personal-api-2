import json
import random

def sample_dataset(input_path, output_path, sample_size, seed=42):
    """
    JSON 데이터셋에서 지정된 개수만큼 무작위 샘플링하여 저장하는 함수
    """
    # 1. 재현성을 위해 시드 설정 (항상 같은 샘플을 뽑으려면 고정)
    random.seed(seed)
    
    print(f"Loading data from {input_path}...")
    
    # 2. 데이터 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 파일을 찾을 수 없습니다.")
        return

    total_len = len(data)
    print(f"Total examples: {total_len}")

    # 3. 샘플링 개수 검증
    if sample_size > total_len:
        print(f"Warning: 요청한 샘플 수({sample_size})가 전체 데이터 수({total_len})보다 큽니다. 전체를 저장합니다.")
        sample_size = total_len

    # 4. 무작위 샘플링 (중복 없이 추출)
    sampled_data = random.sample(data, sample_size)
    print(f"Sampled {len(sampled_data)} examples.")

    # 5. 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        # indent=2는 가독성을 위해 들여쓰기를 함
        # ensure_ascii=False는 한글 등 유니코드가 깨지지 않고 그대로 보이게 함
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")

# 사용 예시
if __name__ == "__main__":
    INPUT_FILE = "sgd_converted_dev_mapped_grouped_with_pref_with_constraints.json"  # 원본 파일명
    OUTPUT_FILE = "sampled_dataset_100.json"  # 저장할 파일명
    NUM_SAMPLES = 100                      # 추출할 샘플 개수
    
    sample_dataset(INPUT_FILE, OUTPUT_FILE, NUM_SAMPLES)