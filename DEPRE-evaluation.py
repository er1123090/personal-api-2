import json
import re
import sys
from collections import Counter
import argparse

def normalize_value(value):
    """값 비교를 위해 문자열로 변환하고 양끝 공백을 제거합니다."""
    if value is None:
        return ""
    return str(value).strip()

def parse_api_string(api_string):
    """API 문자열을 파싱하여 (함수명, 슬롯_딕셔너리) 리스트로 반환합니다."""
    if not api_string:
        return []
    
    parsed_calls = []

    # 1. Regex 파싱
    calls = re.findall(r'(\w+)\((.*?)\)', api_string)
    if calls:
        for func_name, args_str in calls:
            slot_dict = {}
            slots = re.findall(r'(\w+)\s*=\s*(?:["\'](.*?)["\']|([^,\s)\]\'"]+))', args_str)
            for key, val_quoted, val_unquoted in slots:
                raw_val = val_quoted if val_quoted else val_unquoted
                slot_dict[key] = normalize_value(raw_val)
            parsed_calls.append((func_name, slot_dict))
        return parsed_calls

    # 2. JSON 파싱 시도 (Fallback)
    try:
        clean_str = api_string.strip()
        if clean_str.startswith("```"):
            lines = clean_str.splitlines()
            if len(lines) >= 3:
                clean_str = "\n".join(lines[1:-1])
        data = json.loads(clean_str)
        if isinstance(data, dict):
            slot_dict = {str(k): normalize_value(v) for k, v in data.items()}
            parsed_calls.append(("__JSON_PARSED__", slot_dict))
    except (json.JSONDecodeError, Exception):
        pass
        
    return parsed_calls

def calculate_metrics(tp, fp, fn):
    """TP, FP, FN을 기반으로 Precision, Recall, F1을 계산합니다."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def filter_triples_by_preference(triples, preference_map):
    """
    (Function, Slot, Value) 리스트에서 Preference List에 있는 슬롯만 남깁니다.
    preference_map에 함수 자체가 없거나, 해당 슬롯이 없으면 제거됩니다.
    """
    if not preference_map:
        return [] # preference 파일이 있지만 비어있거나 매칭 안되면 빈 리스트 반환 (엄격 적용 시)
        # 만약 preference 파일이 없을 때 필터링 안하려면 input 단계에서 처리

    filtered = []
    for func, slot, val in triples:
        # 1. 해당 함수가 preference 목록에 있는지 확인
        if func in preference_map:
            # 2. 해당 슬롯이 관심 대상 리스트에 있는지 확인
            if slot in preference_map[func]:
                filtered.append((func, slot, val))
    return filtered

def evaluate_dialogues(data, preference_map=None):
    """데이터 리스트를 받아 도메인, 전체 슬롯, 중요 슬롯(Preference) 지표를 계산합니다."""
    
    # 1. Domain Counters
    d_tp, d_fp, d_fn = 0, 0, 0
    d_correct_samples = 0
    
    # 2. All Slots Counters (기존)
    s_tp, s_fp, s_fn = 0, 0, 0
    
    # 3. Preferred Slots Counters (신규 추가)
    ps_tp, ps_fp, ps_fn = 0, 0, 0
    
    valid_samples = 0

    for item in data:
        gt_str = str(item.get("reference_ground_truth", "") or "").strip()
        pred_str = str(item.get("llm_output", "") or "").strip()

        if not gt_str:
            continue
            
        valid_samples += 1
        
        # 파싱 수행
        gt_parsed = parse_api_string(gt_str)
        pred_parsed = parse_api_string(pred_str)
        
        # JSON 후처리 (Heuristic Match)
        if len(pred_parsed) == 1 and pred_parsed[0][0] == "__JSON_PARSED__":
            if len(gt_parsed) == 1:
                guessed_func_name = gt_parsed[0][0]
                pred_slots = pred_parsed[0][1]
                pred_parsed = [(guessed_func_name, pred_slots)]
        
        # ---------------------------------------------------------
        # [Metric 1] Domain (Function Name)
        # ---------------------------------------------------------
        gt_domains = Counter([x[0] for x in gt_parsed])
        pred_domains = Counter([x[0] for x in pred_parsed])
        
        intersection = gt_domains & pred_domains
        d_tp += sum(intersection.values())
        d_fp += sum((pred_domains - gt_domains).values())
        d_fn += sum((gt_domains - pred_domains).values())
        
        if gt_domains == pred_domains:
            d_correct_samples += 1

        # ---------------------------------------------------------
        # [Metric 2] All Slot-Value Pairs (기존 방식)
        # ---------------------------------------------------------
        gt_triples = []
        for func, slots in gt_parsed:
            for s_key, s_val in slots.items():
                gt_triples.append((func, s_key, s_val))
                
        pred_triples = []
        for func, slots in pred_parsed:
            for s_key, s_val in slots.items():
                pred_triples.append((func, s_key, s_val))
                
        gt_slots_counter = Counter(gt_triples)
        pred_slots_counter = Counter(pred_triples)
        
        s_intersect = gt_slots_counter & pred_slots_counter
        s_tp += sum(s_intersect.values())
        s_fp += sum((pred_slots_counter - gt_slots_counter).values())
        s_fn += sum((gt_slots_counter - pred_slots_counter).values())

        # ---------------------------------------------------------
        # [Metric 3] Preferred Slot-Value Pairs (신규 추가)
        # ---------------------------------------------------------
        if preference_map:
            # GT와 Pred 모두에서 "중요하지 않은 슬롯"을 제거합니다.
            gt_pref_triples = filter_triples_by_preference(gt_triples, preference_map)
            pred_pref_triples = filter_triples_by_preference(pred_triples, preference_map)
            
            gt_pref_counter = Counter(gt_pref_triples)
            pred_pref_counter = Counter(pred_pref_triples)
            
            ps_intersect = gt_pref_counter & pred_pref_counter
            ps_tp += sum(ps_intersect.values())
            ps_fp += sum((pred_pref_counter - gt_pref_counter).values())
            ps_fn += sum((gt_pref_counter - pred_pref_counter).values())

    # Metrics Calculation
    d_prec, d_rec, d_f1 = calculate_metrics(d_tp, d_fp, d_fn)
    d_acc = d_correct_samples / valid_samples if valid_samples > 0 else 0
    
    s_prec, s_rec, s_f1 = calculate_metrics(s_tp, s_fp, s_fn)
    
    # Preferred Metrics 초기화
    ps_prec, ps_rec, ps_f1 = 0, 0, 0
    if preference_map:
        ps_prec, ps_rec, ps_f1 = calculate_metrics(ps_tp, ps_fp, ps_fn)
    
    print(f"DEBUG: All TP={s_tp}, FN={s_fn} | Pref TP={ps_tp}, FN={ps_fn}")
    
    return {
        "valid_samples": valid_samples,
        "domain": {"acc": d_acc, "f1": d_f1},
        "all_slots": {"prec": s_prec, "rec": s_rec, "f1": s_f1},
        "pref_slots": {"prec": ps_prec, "rec": ps_rec, "f1": ps_f1}
    }

def run_evaluation(test_file, pref_file=None):
    print(f"Processing data file: {test_file}")
    
    preference_map = None
    if pref_file:
        print(f"Loading preference list from: {pref_file}")
        try:
            with open(pref_file, 'r', encoding='utf-8') as f:
                preference_map = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load preference file. {e}")
            
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"-> Successfully loaded {len(data)} items.")

        results = evaluate_dialogues(data, preference_map)

        print("\n" + "=" * 60)
        print(f" EVALUATION RESULTS (Valid Samples: {results['valid_samples']})")
        print("=" * 60)
        
        print(f"\n[1] Domain (Function Name)")
        print(f" - Accuracy  : {results['domain']['acc']:.4f}")
        print(f" - F1 Score  : {results['domain']['f1']:.4f}")
        
        print(f"\n[2] All Slot-Value Pairs (General)")
        print(f" - F1 Score  : {results['all_slots']['f1']:.4f}")
        print(f" - Precision : {results['all_slots']['prec']:.4f}")
        print(f" - Recall    : {results['all_slots']['rec']:.4f}")

        if preference_map:
            print(f"\n[3] Preferred Slot-Value Pairs (Targeted)")
            print(f" * Only evaluating slots defined in preference_list.json")
            print(f" - F1 Score  : {results['pref_slots']['f1']:.4f}")
            print(f" - Precision : {results['pref_slots']['prec']:.4f}")
            print(f" - Recall    : {results['pref_slots']['rec']:.4f}")
            
        print("=" * 60)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="test_data.json", help="Path to the model output json")
    parser.add_argument("--pref_file", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json", help="Path to the preference list json")
    args = parser.parse_args()
        
    run_evaluation(args.test_file, args.pref_file)