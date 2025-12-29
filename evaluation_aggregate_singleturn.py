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
      - "{GetX}(a=1)" (Curly braces support)
      - JSON dict:
          1) {"GetX": {"a": 1}} -> ("GetX", {"a": "1"})
          2) {"slot": "val"}    -> ("__JSON_PARSED__", {"slot": "val"})
    Returns list of (func_name, slot_dict).
    """
    if api_string is None:
        return []

    # If it's already a dict, treat as JSON-parsed immediately
    if isinstance(api_string, dict):
        return _parse_json_dict(api_string)

    s = str(api_string).strip()
    if not s:
        return []

    parsed_calls: List[Tuple[str, Dict[str, str]]] = []

    # -------------------------------------------------------------------------
    # 1) Regex parse function calls (Text format)
    # -------------------------------------------------------------------------
    # Modified Regex to handle optional curly braces: {GetRideSharing}(...) or GetRideSharing(...)
    calls = re.findall(r'(?:\{?(\w+)\}?)\s*\((.*?)\)', s)
    if calls:
        for func_name, args_str in calls:
            slot_dict = _parse_call_args(args_str)
            parsed_calls.append((func_name, slot_dict))
        return parsed_calls

    # -------------------------------------------------------------------------
    # 2) JSON parsing fallback (handles ```json ... ```)
    # -------------------------------------------------------------------------
    try:
        clean_str = s
        # Remove markdown code blocks if present
        if "```" in clean_str:
            # Extract content strictly between first ```(json)? and last ```
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, clean_str, re.DOTALL)
            if match:
                clean_str = match.group(1).strip()
            else:
                # Fallback: simple strip if regex fails but ``` exists
                lines = clean_str.splitlines()
                if len(lines) >= 3 and lines[0].startswith("```"):
                    clean_str = "\n".join(lines[1:-1])

        data = json.loads(clean_str)
        
        # Handle list of dicts (e.g. [{"GetX": ...}, ...])
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parsed_calls.extend(_parse_json_dict(item))
            return parsed_calls

        # Handle single dict
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
    if not preference_map:
        return []
    filtered = []
    for func, slot, val in triples:
        if func in preference_map and slot in set(preference_map[func]):
            filtered.append((func, slot, val))
    return filtered


# ==============================================================================
# 2) Robust Loading: JSON / JSONL / dict-wrapped results
# ==============================================================================
def _iter_records_from_loaded_json(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Accepts:
      - list[dict]
      - dict with common wrapper keys
      - single dict record
    """
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return

    if isinstance(obj, dict):
        # common wrapper keys (add more if your pipeline uses others)
        for key in ["data", "examples", "results", "outputs", "items"]:
            v = obj.get(key)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return

        # if dict itself looks like a record
        yield obj
        return

    # otherwise nothing


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
                    # skip malformed line
                    pass
        else:
            obj = json.load(f)
            records.extend(list(_iter_records_from_loaded_json(obj)))
    return records


# ==============================================================================
# 3) Extract GT / Pred from your output-json format
# ==============================================================================
def _join_calls(x: Any) -> str:
    """
    Convert:
      - list[str] -> "a\nb\nc"
      - str -> str
      - None -> ""
      - dict -> dict (handled in parse_api_string)
    """
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join([str(t).strip() for t in x if str(t).strip()])
    return str(x).strip()


def extract_gt_pred(item: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Your example shows:
      item["reference_ground_truth"] = ["GetHotels(...)","GetHotels(...)"]
      item["llm_output"] = "GetHotels(...)"
    """
    gt = item.get("reference_ground_truth", "")
    pred = item.get("llm_output", "")
    return gt, pred


def parse_pred(pred_raw: Any) -> List[Tuple[str, Dict[str, str]]]:
    """
    llm_output can be str / list[str] / dict
    """
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
    """
    Keep your original heuristic:
    if pred parsed as JSON dict, align function name with GT's single function.
    """
    if len(pred_parsed) == 1 and pred_parsed[0][0] == "__JSON_PARSED__":
        if len(gt_parsed) == 1:
            return [(gt_parsed[0][0], pred_parsed[0][1])]
    return pred_parsed


def _count_domain_slot_pref(
    gt_parsed: List[Tuple[str, Dict[str, str]]],
    pred_parsed: List[Tuple[str, Dict[str, str]]],
    preference_map: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    # Domain
    gt_domains = Counter([x[0] for x in gt_parsed])
    pred_domains = Counter([x[0] for x in pred_parsed])

    d_tp = sum((gt_domains & pred_domains).values())
    d_fp = sum((pred_domains - gt_domains).values())
    d_fn = sum((gt_domains - pred_domains).values())
    domain_correct = int(gt_domains == pred_domains)

    # Slot triples
    gt_triples = _triples_from_parsed(gt_parsed)
    pred_triples = _triples_from_parsed(pred_parsed)

    gt_slots_cnt = Counter(gt_triples)
    pred_slots_cnt = Counter(pred_triples)

    s_tp = sum((gt_slots_cnt & pred_slots_cnt).values())
    s_fp = sum((pred_slots_cnt - gt_slots_cnt).values())
    s_fn = sum((gt_slots_cnt - pred_slots_cnt).values())

    # Pref-slot triples
    ps_tp = ps_fp = ps_fn = 0
    if preference_map:
        gt_pref = filter_triples_by_preference(gt_triples, preference_map)
        pred_pref = filter_triples_by_preference(pred_triples, preference_map)

        gt_pref_cnt = Counter(gt_pref)
        pred_pref_cnt = Counter(pred_pref)

        ps_tp = sum((gt_pref_cnt & pred_pref_cnt).values())
        ps_fp = sum((pred_pref_cnt - gt_pref_cnt).values())
        ps_fn = sum((gt_pref_cnt - pred_pref_cnt).values())

    s_prec, s_rec, s_f1 = calculate_metrics(s_tp, s_fp, s_fn)
    d_prec, d_rec, d_f1 = calculate_metrics(d_tp, d_fp, d_fn)
    ps_prec, ps_rec, ps_f1 = calculate_metrics(ps_tp, ps_fp, ps_fn) if preference_map else (0.0, 0.0, 0.0)

    return {
        "d_tp": d_tp, "d_fp": d_fp, "d_fn": d_fn, "domain_correct": domain_correct, "domain_f1": d_f1,
        "s_tp": s_tp, "s_fp": s_fp, "s_fn": s_fn, "slot_f1": s_f1, "slot_fp": s_fp,
        "ps_tp": ps_tp, "ps_fp": ps_fp, "ps_fn": ps_fn, "pref_slot_f1": ps_f1, "pref_slot_fp": ps_fp,
    }


def parse_gt_candidates(gt_raw: Any) -> List[List[Tuple[str, Dict[str, str]]]]:
    """
    OR semantics:
      - if gt_raw is list[str], treat each entry as an alternative GT candidate
      - if gt_raw is str, single candidate
    """
    if gt_raw is None:
        return []

    if isinstance(gt_raw, list):
        candidates: List[List[Tuple[str, Dict[str, str]]]] = []
        for g in gt_raw:
            g_str = _join_calls(g)
            parsed = parse_api_string(g_str)
            if parsed:
                candidates.append(parsed)
        return candidates

    # single string/dict
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

    Primary score:
      - if preference_map exists: pref_slot_f1
      - else: slot_f1

    Tie-breaks:
      - slot_f1
      - domain_f1
      - fewer FP (pref_slot_fp or slot_fp)
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

        key = (primary, stats["slot_f1"], stats["domain_f1"], -fp_penalty)

        if best_stats is None or (best_key is not None and key > best_key):
            best_stats = stats
            best_key = key

    return best_stats


# ==============================================================================
# 5) Single-file evaluation
# ==============================================================================
def evaluate_output_file(file_path: str, preference_map: Optional[Dict[str, List[str]]] = None) -> Optional[Dict[str, Any]]:
    try:
        data = load_records(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

    d_tp = d_fp = d_fn = 0
    d_correct = 0

    s_tp = s_fp = s_fn = 0
    ps_tp = ps_fp = ps_fn = 0

    valid_samples = 0
    parse_success_cnt = 0
    parse_fail_cnt = 0

    for item in data:
        gt_raw, pred_raw = extract_gt_pred(item)

        # skip if GT missing
        if gt_raw is None or (isinstance(gt_raw, str) and not gt_raw.strip()) or (isinstance(gt_raw, list) and len(gt_raw) == 0):
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
            continue

        # Accumulate using the chosen (best-matching) GT candidate
        d_tp += best["d_tp"]; d_fp += best["d_fp"]; d_fn += best["d_fn"]
        d_correct += best["domain_correct"]

        s_tp += best["s_tp"]; s_fp += best["s_fp"]; s_fn += best["s_fn"]

        if preference_map:
            ps_tp += best["ps_tp"]; ps_fp += best["ps_fp"]; ps_fn += best["ps_fn"]

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
    Optional: keep your path-based metadata extraction.
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