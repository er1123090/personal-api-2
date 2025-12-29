import os
import json
import argparse
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm


# ==============================================================================
# 1) Parsing & Metrics
# ==============================================================================
def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_call_args(args_str: str) -> Dict[str, str]:
    """
    Parse key=value pairs inside func_name(...)
    Supports quoted/unquoted values.
    """
    slot_dict: Dict[str, str] = {}
    # key="abc" or key='abc' or key=abc
    # Regex captures: 1=key, 2=val_quoted, 3=val_unquoted
    slots = re.findall(r'(\w+)\s*=\s*(?:["\'](.*?)["\']|([^,\s)\]\'"]+))', args_str)
    for key, val_quoted, val_unquoted in slots:
        raw_val = val_quoted if val_quoted else val_unquoted
        slot_dict[key] = normalize_value(raw_val)
    return slot_dict


def _parse_json_dict(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, str]]]:
    """
    Helper to distinguish between:
    A) {"FuncName": {"arg": "val"}} -> Specific function call
    B) {"arg": "val"} -> Generic JSON parsed (fallback)
    """
    results = []
    
    # Check if this dict represents specific function calls (Key=Func, Value=Dict)
    # Heuristic: If ANY value is a dictionary, assume structure A.
    is_nested_func = False
    for v in data.values():
        if isinstance(v, dict):
            is_nested_func = True
            break
            
    if is_nested_func:
        # Structure A: {"GetFlights": {"dest": "Nairobi"}, "GetHotels": {...}}
        for func_name, args in data.items():
            if isinstance(args, dict):
                slot_dict = {str(k): normalize_value(v) for k, v in args.items()}
                results.append((func_name, slot_dict))
    else:
        # Structure B: {"destination": "Nairobi", "passengers": "2"}
        # No function name inferred, use generic tag.
        slot_dict = {str(k): normalize_value(v) for k, v in data.items()}
        results.append(("__JSON_PARSED__", slot_dict))
        
    return results


def parse_api_string(api_string: Any) -> List[Tuple[str, Dict[str, str]]]:
    """
    Parse:
      - "GetX(a=1) GetY(b='c')"
      - "{GetX}(a=1)" (Curly braces support added)
      - JSON dict formats
    Returns list of (func_name, slot_dict).
    """
    if api_string is None:
        return []

    # If it's already a dict/list, treat as JSON-parsed immediately
    if isinstance(api_string, (dict, list)):
        if isinstance(api_string, list):
            # Handle list of dicts directly provided as object
            parsed_list = []
            for item in api_string:
                if isinstance(item, dict):
                    parsed_list.extend(_parse_json_dict(item))
            return parsed_list
        return _parse_json_dict(api_string)

    s = str(api_string).strip()
    if not s:
        return []

    parsed_calls: List[Tuple[str, Dict[str, str]]] = []

    # -------------------------------------------------------------------------
    # 1) Regex parse function calls (Text format)
    # -------------------------------------------------------------------------
    # Modified Regex to handle optional curly braces: {GetRideSharing}(...) or GetRideSharing(...)
    pattern = r'(?:\{?(\w+)\}?)\s*\((.*?)\)'
    calls = re.findall(pattern, s, re.DOTALL)
    
    if calls:
        for func_name, args_str in calls:
            # Clean up function name just in case regex captured braces
            clean_func = func_name.replace("{", "").replace("}", "")
            slot_dict = _parse_call_args(args_str)
            parsed_calls.append((clean_func, slot_dict))
        return parsed_calls

    # -------------------------------------------------------------------------
    # 2) JSON parsing fallback (handles ```json ... ```)
    # -------------------------------------------------------------------------
    try:
        clean_str = s
        # Remove markdown code blocks if present
        if "```" in clean_str:
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, clean_str, re.DOTALL)
            if match:
                clean_str = match.group(1).strip()
            else:
                lines = clean_str.splitlines()
                if len(lines) >= 3 and lines[0].startswith("```"):
                    clean_str = "\n".join(lines[1:-1])

        data = json.loads(clean_str)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parsed_calls.extend(_parse_json_dict(item))
            return parsed_calls

        if isinstance(data, dict):
            parsed_calls.extend(_parse_json_dict(data))
            return parsed_calls

    except Exception:
        pass

    return []


def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def filter_triples_by_preference(
    triples: List[Tuple[str, str, str]],
    preference_map: Dict[str, List[str]]
) -> List[Tuple[str, str, str]]:
    """
    [핵심 필터링]
    오직 preference_map에 등록된 (함수, 슬롯) 조합만 남기고 나머지는 제거합니다.
    """
    if not preference_map:
        return []
    filtered = []
    for func, slot, val in triples:
        # map에 해당 함수 키가 있고, 그 리스트 안에 해당 슬롯이 있어야 함
        if func in preference_map and slot in set(preference_map[func]):
            filtered.append((func, slot, val))
    return filtered


# ==============================================================================
# 2) Robust Loading: JSON / JSONL / dict-wrapped results
# ==============================================================================
def _iter_records_from_loaded_json(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return

    if isinstance(obj, dict):
        for key in ["data", "examples", "results", "outputs", "items"]:
            v = obj.get(key)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return
        yield obj
        return


def load_records(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.extend(list(_iter_records_from_loaded_json(obj)))
                except Exception:
                    pass
        else:
            obj = json.load(f)
            records.extend(list(_iter_records_from_loaded_json(obj)))
    return records


# ==============================================================================
# 3) Extract GT / Pred from your output-json format
# ==============================================================================
def _join_calls(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join([str(t).strip() for t in x if str(t).strip()])
    return str(x).strip()


def extract_gt_pred(item: Dict[str, Any]) -> Tuple[Any, Any]:
    gt = item.get("reference_ground_truth", "")
    pred = item.get("llm_output", "")
    return gt, pred


def parse_pred(pred_raw: Any) -> List[Tuple[str, Dict[str, str]]]:
    if pred_raw is None:
        return []
    if isinstance(pred_raw, list):
        pred_joined = _join_calls(pred_raw)
        return parse_api_string(pred_joined)
    return parse_api_string(pred_raw)


# ==============================================================================
# 4) OR-semantics for GT list (pick best-matching GT candidate)
# ==============================================================================
def _triples_from_parsed(parsed_calls: List[Tuple[str, Dict[str, str]]]) -> List[Tuple[str, str, str]]:
    return [(f, k, v) for f, s in parsed_calls for k, v in s.items()]


def align_json_pred_to_gt(
    pred_parsed: List[Tuple[str, Dict[str, str]]],
    gt_parsed: List[Tuple[str, Dict[str, str]]],
) -> List[Tuple[str, Dict[str, str]]]:
    if len(pred_parsed) == 1 and pred_parsed[0][0] == "__JSON_PARSED__":
        if len(gt_parsed) == 1:
            return [(gt_parsed[0][0], pred_parsed[0][1])]
    return pred_parsed


def _count_domain_slot_pref(
    gt_parsed: List[Tuple[str, Dict[str, str]]],
    pred_parsed: List[Tuple[str, Dict[str, str]]],
    preference_map: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    
    # 1. Domain (함수명) 계산 - (필터링 전 원본 기준)
    gt_domains = Counter([x[0] for x in gt_parsed])
    pred_domains = Counter([x[0] for x in pred_parsed])

    d_tp = sum((gt_domains & pred_domains).values())
    d_fp = sum((pred_domains - gt_domains).values())
    d_fn = sum((gt_domains - pred_domains).values())
    domain_correct = int(gt_domains == pred_domains)
    d_prec, d_rec, d_f1 = calculate_metrics(d_tp, d_fp, d_fn)

    # 2. All Slot (전체 슬롯) 계산 - (필터링 전 원본 기준, 참고용)
    gt_triples = _triples_from_parsed(gt_parsed)
    pred_triples = _triples_from_parsed(pred_parsed)

    gt_slots_cnt = Counter(gt_triples)
    pred_slots_cnt = Counter(pred_triples)

    s_tp = sum((gt_slots_cnt & pred_slots_cnt).values())
    s_fp = sum((pred_slots_cnt - gt_slots_cnt).values())
    s_fn = sum((gt_slots_cnt - pred_slots_cnt).values())
    s_prec, s_rec, s_f1 = calculate_metrics(s_tp, s_fp, s_fn)

    # 3. [중요] Preference Slot 계산 (엄격한 필터링 적용)
    # pref_list에 없는 슬롯은 여기서 모두 제거되어 계산에 반영되지 않음
    ps_tp = ps_fp = ps_fn = 0
    ps_prec = ps_rec = ps_f1 = 0.0

    if preference_map:
        # A. 필터링: pref_list에 없는 슬롯은 제거됨
        gt_pref = filter_triples_by_preference(gt_triples, preference_map)
        pred_pref = filter_triples_by_preference(pred_triples, preference_map)

        # B. 카운팅: 오직 필터링된 리스트만 가지고 비교
        gt_pref_cnt = Counter(gt_pref)
        pred_pref_cnt = Counter(pred_pref)

        # C. 교집합(TP), 차집합(FP, FN) 계산
        ps_tp = sum((gt_pref_cnt & pred_pref_cnt).values())
        ps_fp = sum((pred_pref_cnt - gt_pref_cnt).values()) # 필터링 후 남은 것 중 Pred에만 있는 것 (오탐)
        ps_fn = sum((gt_pref_cnt - pred_pref_cnt).values()) # 필터링 후 남은 것 중 GT에만 있는 것 (미탐)

        # D. 점수 계산
        ps_prec, ps_rec, ps_f1 = calculate_metrics(ps_tp, ps_fp, ps_fn)

    return {
        # Domain Metrics
        "d_tp": d_tp, "d_fp": d_fp, "d_fn": d_fn, "domain_correct": domain_correct, "domain_f1": d_f1,
        # All Slot Metrics
        "s_tp": s_tp, "s_fp": s_fp, "s_fn": s_fn, "slot_f1": s_f1, "slot_fp": s_fp,
        # Pref Slot Metrics (Filtered)
        "ps_tp": ps_tp, "ps_fp": ps_fp, "ps_fn": ps_fn, "pref_slot_f1": ps_f1, "pref_slot_fp": ps_fp,
    }


def parse_gt_candidates(gt_raw: Any) -> List[List[Tuple[str, Dict[str, str]]]]:
    """
    OR semantics:
      - if gt_raw is list, treat each item as an independent alternative GT candidate.
      - if gt_raw is str/dict, treat as single candidate.
    """
    if gt_raw is None:
        return []

    # If list, treat each element as a separate valid ground truth (OR logic)
    if isinstance(gt_raw, list):
        candidates: List[List[Tuple[str, Dict[str, str]]]] = []
        for g in gt_raw:
            g_str = _join_calls(g)
            parsed = parse_api_string(g_str)
            if parsed:
                candidates.append(parsed)
        return candidates

    # Single string/dict
    g_str = _join_calls(gt_raw)
    parsed = parse_api_string(g_str)
    return [parsed] if parsed or g_str.strip() else []


def choose_best_gt_candidate(
    gt_candidates: List[List[Tuple[str, Dict[str, str]]]],
    pred_parsed_raw: List[Tuple[str, Dict[str, str]]],
    preference_map: Optional[Dict[str, List[str]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate pred against each GT candidate and pick the best one.

    Primary score hierarchy:
      1. Preference F1 (FILTERED) -> 가장 중요
      2. Slot F1 (Unfiltered)
      3. Domain F1
      4. False Positives (Lower is better)
    """
    if not gt_candidates:
        return None

    best_stats: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, int]] = None

    for gt_parsed in gt_candidates:
        pred_parsed = align_json_pred_to_gt(pred_parsed_raw, gt_parsed)
        stats = _count_domain_slot_pref(gt_parsed, pred_parsed, preference_map)

        primary = stats["pref_slot_f1"] if preference_map else stats["slot_f1"]
        fp_penalty = stats["pref_slot_fp"] if preference_map else stats["slot_fp"]

        # Score key: (primary_f1, slot_f1, domain_f1, -fp_penalty)
        key = (primary, stats["slot_f1"], stats["domain_f1"], -fp_penalty)

        if best_stats is None or (best_key is not None and key > best_key):
            best_stats = stats
            best_key = key

    return best_stats


# ==============================================================================
# 5) Single-file evaluation (Modified)
# ==============================================================================
def _save_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    try:
        # JSONL Format
        if file_path.endswith(".jsonl"):
            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # JSON Format (List of Dicts)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to save {file_path}: {e}")


def evaluate_output_file(file_path: str, preference_map: Optional[Dict[str, List[str]]] = None) -> Optional[Dict[str, Any]]:
    try:
        data = load_records(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

    d_tp = d_fp = d_fn = d_correct = 0
    s_tp = s_fp = s_fn = 0
    ps_tp = ps_fp = ps_fn = 0

    valid_samples = 0
    parse_success_cnt = 0
    parse_fail_cnt = 0

    # --------------------------------------------------------------------------
    # Loop for evaluation & marking 'evaluation_result'
    # --------------------------------------------------------------------------
    for item in data:
        gt_raw, pred_raw = extract_gt_pred(item)

        # skip if GT missing
        if gt_raw is None or (isinstance(gt_raw, str) and not gt_raw.strip()) or (isinstance(gt_raw, list) and len(gt_raw) == 0):
            item["evaluation_result"] = False
            continue

        valid_samples += 1

        # Parse pred once
        pred_parsed_raw = parse_pred(pred_raw)
        if len(pred_parsed_raw) > 0:
            parse_success_cnt += 1
        else:
            parse_fail_cnt += 1

        # OR-semantics: GT list means "any one of them is acceptable"
        gt_candidates = parse_gt_candidates(gt_raw)
        best = choose_best_gt_candidate(gt_candidates, pred_parsed_raw, preference_map=preference_map)
        
        if best is None:
            # Parsing or matching failed completely
            item["evaluation_result"] = False
            continue

        # ----------------------------------------------------------------------
        # Correctness Check
        # ----------------------------------------------------------------------
        if preference_map:
            # [핵심] pref_list에 정의된 슬롯들에 대해서만!
            # 1. 미탐(FN)이 없어야 함 (필수 선호도 모두 충족)
            # 2. 오탐(FP)이 없어야 함 (선호도 리스트에 있는 슬롯 값을 틀리면 안 됨)
            # * 주의: pref_list에 없는 슬롯은 이미 필터링되어 ps_fp 계산에서 제외됨.
            is_correct = (best["ps_fn"] == 0) and (best["ps_fp"] == 0)
        else:
            # Fallback if no pref map: Strict exact match on all slots & domains
            is_correct = (best["s_fn"] == 0) and (best["s_fp"] == 0) and (best["domain_correct"] == 1)

        item["evaluation_result"] = is_correct

        # Accumulate metrics (using best match stats)
        d_tp += best["d_tp"]; d_fp += best["d_fp"]; d_fn += best["d_fn"]
        d_correct += best["domain_correct"]

        s_tp += best["s_tp"]; s_fp += best["s_fp"]; s_fn += best["s_fn"]

        if preference_map:
            ps_tp += best["ps_tp"]; ps_fp += best["ps_fp"]; ps_fn += best["ps_fn"]

    # --------------------------------------------------------------------------
    # Overwrite the file with updated data (including 'evaluation_result')
    # --------------------------------------------------------------------------
    _save_file(file_path, data)

    # --------------------------------------------------------------------------
    # Calculate Aggregate Metrics
    # --------------------------------------------------------------------------
    d_prec, d_rec, d_f1 = calculate_metrics(d_tp, d_fp, d_fn)
    d_acc = d_correct / valid_samples if valid_samples > 0 else 0.0

    s_prec, s_rec, s_f1 = calculate_metrics(s_tp, s_fp, s_fn)

    ps_prec = ps_rec = ps_f1 = 0.0
    if preference_map:
        ps_prec, ps_rec, ps_f1 = calculate_metrics(ps_tp, ps_fp, ps_fn)

    total_attempts = parse_success_cnt + parse_fail_cnt
    parse_error_rate = (parse_fail_cnt / total_attempts) if total_attempts > 0 else 0.0

    return {
        "file_path": file_path,
        "valid_samples": valid_samples,
        "parse_success": parse_success_cnt,
        "parse_fail": parse_fail_cnt,
        "parse_error_rate": parse_error_rate,
        "domain_acc": d_acc,
        "domain_f1": d_f1,
        "slot_prec": s_prec,
        "slot_rec": s_rec,
        "slot_f1": s_f1,
        "pref_slot_prec": ps_prec,
        "pref_slot_rec": ps_rec,
        "pref_slot_f1": ps_f1,
    }


# ==============================================================================
# 6) Directory aggregation (recursive scan)
# ==============================================================================
def load_preference_map(pref_file: str) -> Optional[Dict[str, List[str]]]:
    if not pref_file:
        return None
    if not os.path.exists(pref_file):
        print(f"[WARN] Preference file not found: {pref_file}")
        return None
    with open(pref_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # expected: { "GetFlights": ["slot1","slot2"], ... }
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, list):
                cleaned[k] = [str(x) for x in v]
            else:
                cleaned[k] = [str(v)]
        return cleaned
    return None


def infer_metadata_from_path(file_path: str, root_dir: str) -> Dict[str, str]:
    """
    Extract metadata from file path structure.
    """
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)

    meta = {
        "context_type": "unknown",
        "difficulty": "unknown",
        "model_name": "unknown",
        "pref_group": "unknown",
        "filename": parts[-1],
        "experiment_group": "unknown",
    }

    # Example structure: context/difficulty/model/[pref_group]/file.jsonl
    if len(parts) >= 3:
        meta["context_type"] = parts[0]
        meta["difficulty"] = parts[1]
        meta["model_name"] = parts[2]
    if len(parts) >= 4:
        meta["pref_group"] = parts[3]

    if "ours_memory" in parts:
        meta["experiment_group"] = "ours_memory"
    elif "mem0" in parts:
        meta["experiment_group"] = "mem0"
    elif "vanillaLLM" in parts:
        meta["experiment_group"] = "vanillaLLM"

    return meta


def aggregate_results(root_dir: str, pref_file: str, output_csv: str) -> None:
    preference_map = load_preference_map(pref_file)

    files_found: List[str] = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith((".json", ".jsonl")):
                files_found.append(os.path.join(r, fn))

    print(f"[INFO] Scanning: {root_dir}")
    print(f"[INFO] Found {len(files_found)} files")

    rows: List[Dict[str, Any]] = []
    for fp in tqdm(files_found, desc="Evaluating"):
        metrics = evaluate_output_file(fp, preference_map=preference_map)
        if not metrics:
            continue

        meta = infer_metadata_from_path(fp, root_dir)
        row = {
            **meta,
            "valid_samples": metrics["valid_samples"],
            "parse_success": metrics["parse_success"],
            "parse_fail": metrics["parse_fail"],
            "parse_error_rate": round(metrics["parse_error_rate"], 4),
            "domain_acc": round(metrics["domain_acc"], 4),
            "domain_f1": round(metrics["domain_f1"], 4),
            "slot_f1": round(metrics["slot_f1"], 4),
            "slot_prec": round(metrics["slot_prec"], 4),
            "slot_rec": round(metrics["slot_rec"], 4),
            "pref_slot_f1": round(metrics["pref_slot_f1"], 4),
            "pref_slot_prec": round(metrics["pref_slot_prec"], 4),
            "pref_slot_rec": round(metrics["pref_slot_rec"], 4),
            "file_path": metrics["file_path"],
        }
        rows.append(row)

    if not rows:
        print("[WARN] No results processed.")
        return

    df = pd.DataFrame(rows)
    sort_cols = ["experiment_group", "context_type", "difficulty", "model_name", "pref_group", "filename"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {output_csv}")

    preview_cols = ["experiment_group", "model_name", "difficulty", "parse_error_rate", "pref_slot_f1", "slot_f1", "domain_acc"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].to_string(index=False))


# ==============================================================================
# 7) CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing output json/jsonl files")
    parser.add_argument("--pref_file", type=str, default="/data/minseo/experiments4/pref_list.json", help="Preference map json (optional)")
    parser.add_argument("--output_csv", type=str, default="aggregated_results.csv", help="Output CSV path")
    args = parser.parse_args()

    aggregate_results(args.root_dir, args.pref_file, args.output_csv)


if __name__ == "__main__":
    main()