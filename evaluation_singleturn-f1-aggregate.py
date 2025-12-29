#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict

# =========================================================
# Robust Parsing (Func(...), {Func}(...), JSON, code fences)
# =========================================================

_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)")
_BRACED_FUNC_RE = re.compile(r"\{([A-Za-z_]\w*)\}\s*\(")  # {GetHotels}(
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s.strip())

def _normalize_braced_func(s: str) -> str:
    return _BRACED_FUNC_RE.sub(r"\1(", s)

def _strip_think_tags(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)
    return s

def _remove_code_fences_keep_content(s: str) -> str:
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")
    return s

def _build_call_string(func: str, args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return f"{func}()"
    parts = []
    for k, v in args.items():
        if v is None:
            continue
        parts.append(f'{k}="{str(v)}"')
    return f"{func}({', '.join(parts)})"

def _json_to_calls(obj: Any) -> List[str]:
    calls: List[str] = []

    if isinstance(obj, list):
        for item in obj:
            calls.extend(_json_to_calls(item))
        return calls

    if isinstance(obj, dict):
        if "calls" in obj and isinstance(obj["calls"], list):
            for c in obj["calls"]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("tool") or c.get("function")
                args = c.get("arguments") or c.get("args") or {}
                if isinstance(name, str):
                    calls.append(_build_call_string(name, args))
            return calls

        if ("name" in obj or "tool" in obj or "function" in obj) and ("arguments" in obj or "args" in obj):
            name = obj.get("name") or obj.get("tool") or obj.get("function")
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(name, str):
                calls.append(_build_call_string(name, args))
            return calls

        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if k in {"reasoning", "raw_content", "model_name", "evaluation_result", "reasoning_tokens"}:
                continue

            if isinstance(v, dict):
                calls.append(_build_call_string(k, v))
            elif v is None:
                calls.append(f"{k}()")
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        calls.append(_build_call_string(k, item))
                    else:
                        calls.append(_build_call_string(k, {"value": item}))
            else:
                calls.append(_build_call_string(k, {"value": v}))
        return calls

    return calls

def _try_parse_json_to_calls(s: str) -> List[str]:
    try:
        obj = json.loads(s)
    except Exception:
        return []
    return _json_to_calls(obj)

def extract_calls(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if isinstance(item, str) and item.strip():
                out.extend(extract_calls(item))
        return out

    if not isinstance(x, str):
        return []

    s = x.strip()
    if not s:
        return []

    s = _strip_think_tags(s)
    s = _remove_code_fences_keep_content(s)
    s = _strip_code_fences(s)
    s = _normalize_braced_func(s)

    json_calls = _try_parse_json_to_calls(s)
    if json_calls:
        return json_calls

    calls = [m.group(0) for m in _CALL_RE.finditer(s)]
    if not calls and _CALL_RE.fullmatch(s):
        calls = [s]
    return calls

# =========================================================
# Slot/Value parsing inside Func(...)
# =========================================================

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def _split_args(arg_str: str) -> List[str]:
    parts, buf = [], []
    in_quote: Optional[str] = None
    escape = False
    for ch in arg_str:
        if escape:
            buf.append(ch); escape = False; continue
        if ch == "\\":
            buf.append(ch); escape = True; continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

def parse_call_to_slotvals(call: str) -> List[Tuple[str, str, str]]:
    call = call.strip()
    m = _CALL_RE.fullmatch(call)
    if not m:
        return []
    domain = m.group(1).strip()
    args_str = m.group(2).strip()
    if not args_str:
        return []
    out: List[Tuple[str, str, str]] = []
    for part in _split_args(args_str):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        slot = k.strip()
        val = _strip_quotes(v.strip())
        out.append((domain, slot, val))
    return out

# =========================================================
# Metric: (domain,slot) AND; value OR within same (domain,slot)
# =========================================================

@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return PRF(p, r, f1, tp, fp, fn)

def build_gt_allowed_map(gt_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    allowed = defaultdict(set)
    for call in extract_calls(gt_field):
        for d, s, v in parse_call_to_slotvals(call):
            allowed[(d, s)].add(v)
    return dict(allowed)

def build_pred_map(pred_field: Any) -> Dict[Tuple[str, str], Set[str]]:
    pred = defaultdict(set)
    for call in extract_calls(pred_field):
        for d, s, v in parse_call_to_slotvals(call):
            pred[(d, s)].add(v)
    return dict(pred)

def counts_slot_and_value_or(
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0

    for key, allowed_vals in gt_allowed.items():
        pv = pred_vals.get(key, set())
        if pv and (pv & allowed_vals):
            tp += 1
        else:
            fn += 1

    for key, pv in pred_vals.items():
        if key not in gt_allowed:
            fp += 1
        else:
            allowed_vals = gt_allowed[key]
            if not (pv & allowed_vals):
                fp += 1

    return tp, fp, fn

def micro_f1_slot_and_value_or(
    examples: List[Dict[str, Any]],
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> PRF:
    TP = FP = FN = 0
    for ex in examples:
        gt_allowed = build_gt_allowed_map(ex.get(gt_key))
        pred_vals = build_pred_map(ex.get(pred_key))
        tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
        TP += tp; FP += fp; FN += fn
    return prf_from_counts(TP, FP, FN)

# =========================================================
# OR-case logging
# =========================================================

def format_or_case_block(
    ex: Dict[str, Any],
    gt_allowed: Dict[Tuple[str, str], Set[str]],
    pred_vals: Dict[Tuple[str, str], Set[str]],
    tp: int, fp: int, fn: int
) -> str:
    or_slots = {k: v for k, v in gt_allowed.items() if len(v) >= 2}
    if not or_slots:
        return ""

    ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
    utt = ex.get("user_utterance", "")
    gt_raw = ex.get("reference_ground_truth", None)
    pred_raw = ex.get("llm_output", None)
    m = prf_from_counts(tp, fp, fn)

    lines = []
    lines.append("=" * 80)
    lines.append(f"EXAMPLE: {ex_id}")
    if utt:
        lines.append(f"USER_UTTERANCE: {utt}")
    lines.append(f"GT_RAW: {gt_raw}")
    lines.append(f"PRED_RAW: {pred_raw}")
    lines.append("")
    lines.append("OR SLOTS (same domain+slot, multiple acceptable values):")
    for (d, s), allowed_vals in sorted(or_slots.items(), key=lambda x: (x[0][0], x[0][1])):
        pv = pred_vals.get((d, s), set())
        hit = bool(pv & allowed_vals)
        lines.append(f"- ({d}, {s}) allowed={sorted(list(allowed_vals))}")
        lines.append(f"  pred={sorted(list(pv)) if pv else []}  -> {'HIT' if hit else 'MISS'}")

    lines.append("")
    lines.append(f"COUNTS: TP={tp} FP={fp} FN={fn}")
    lines.append(f"PRF: P={m.precision:.4f} R={m.recall:.4f} F1={m.f1:.4f}")
    lines.append("")
    return "\n".join(lines)

def write_or_case_log(
    examples: List[Dict[str, Any]],
    out_path: str,
    gt_key: str = "reference_ground_truth",
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        wrote_any = False
        for ex in examples:
            gt_allowed = build_gt_allowed_map(ex.get(gt_key))
            pred_vals = build_pred_map(ex.get(pred_key))
            tp, fp, fn = counts_slot_and_value_or(gt_allowed, pred_vals)
            block = format_or_case_block(ex, gt_allowed, pred_vals, tp, fp, fn)
            if block:
                f.write(block)
                f.write("\n")
                wrote_any = True

        if not wrote_any:
            f.write("No OR cases found (no (domain,slot) with >=2 acceptable values).\n")

# =========================================================
# Parsing-failure logging
# =========================================================

def is_parsing_failed_pred(ex: Dict[str, Any], pred_key: str = "llm_output") -> Tuple[bool, str]:
    pred_raw = ex.get(pred_key)

    calls = extract_calls(pred_raw)
    if not calls:
        return True, "no_calls_extracted"

    pred_map = build_pred_map(pred_raw)
    if not pred_map:
        return True, "no_slotvals_parsed"

    return False, "ok"

def write_parsing_failures(
    examples: List[Dict[str, Any]],
    out_path: str,
    pred_key: str = "llm_output",
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        n_fail = 0
        for ex in examples:
            failed, reason = is_parsing_failed_pred(ex, pred_key=pred_key)
            if failed:
                ex_id = ex.get("example_id_sub") or ex.get("example_id") or "NA"
                f.write(f"{ex_id}\t{reason}\n")
                n_fail += 1
        f.write(f"\n# total_failures={n_fail} / total_examples={len(examples)}\n")

# =========================================================
# JSON loading helpers
# =========================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_examples(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported JSON structure: expected list or dict with data/examples/items list.")

# =========================================================
# Evaluate one file
# =========================================================

def eval_one_file(
    json_path: str,
    or_log_path: Optional[str] = None,
    parsing_fail_path: Optional[str] = None,
) -> PRF:
    data = load_json(json_path)
    examples = extract_examples(data)

    overall = micro_f1_slot_and_value_or(examples)

    if or_log_path:
        os.makedirs(os.path.dirname(or_log_path), exist_ok=True)
        write_or_case_log(examples, or_log_path)

    if parsing_fail_path:
        os.makedirs(os.path.dirname(parsing_fail_path), exist_ok=True)
        write_parsing_failures(examples, parsing_fail_path)

    return overall

# =========================================================
# Evaluate all json under root_dir and save CSV
# Folder schema:
#   root_dir/context/pref_type/query_turns/model_name/prompt_type/file.json
# =========================================================

def iter_json_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def parse_hparams_from_relpath(rel_path: str) -> Dict[str, str]:
    """
    Expect: context/pref_type/query_turns/model_name/prompt_type/file.json
    If shorter/longer, fill missing as "" and keep best-effort.
    """
    parts = rel_path.split(os.sep)
    # take last 6 components if longer (to be robust)
    if len(parts) >= 6:
        context, pref_type, query_turns, model_name, prompt_type, file_name = parts[-6:]
    else:
        # pad from left
        padded = [""] * (6 - len(parts)) + parts
        context, pref_type, query_turns, model_name, prompt_type, file_name = padded[-6:]
    return {
        "context": context,
        "pref_type": pref_type,
        "query_turns": query_turns,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "file_name": file_name,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Recursively evaluate all .json files under this directory.")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Evaluate a single json file (ignored if --root_dir is set).")

    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path (per-file metrics).")

    parser.add_argument("--save_or_logs", action="store_true",
                        help="Save OR-case logs per json file.")
    parser.add_argument("--save_parsing_failures", action="store_true",
                        help="Save parsing-failure example_id_sub list per json file.")

    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Directory to place logs. Default: alongside out_csv in <out_csv_dir>/logs")

    args = parser.parse_args()

    if args.root_dir:
        targets = iter_json_files(args.root_dir)
        if not targets:
            raise FileNotFoundError(f"No .json files found under root_dir={args.root_dir}")
    elif args.json_path:
        targets = [args.json_path]
    else:
        raise ValueError("Provide either --root_dir or --json_path")

    out_csv = args.out_csv
    ensure_parent_dir(out_csv)

    logs_dir = args.logs_dir or os.path.join(os.path.dirname(out_csv) or ".", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    rows = []
    for jp in targets:
        if args.root_dir:
            rel = os.path.relpath(jp, args.root_dir)
            h = parse_hparams_from_relpath(rel)
            # group key based on folder structure (no file)
            group_key = os.path.join(h["context"], h["pref_type"], h["query_turns"], h["model_name"], h["prompt_type"])
        else:
            rel = os.path.basename(jp)
            h = {"context": "", "pref_type": "", "query_turns": "", "model_name": "", "prompt_type": "", "file_name": os.path.basename(jp)}
            group_key = ""

        stem = rel.replace(os.sep, "__")  # safe filename for logs

        or_log_path = None
        parsing_fail_path = None
        if args.save_or_logs:
            or_log_path = os.path.join(logs_dir, f"{stem}.or_cases.txt")
        if args.save_parsing_failures:
            parsing_fail_path = os.path.join(logs_dir, f"{stem}.parsing_failures.txt")

        try:
            prf = eval_one_file(jp, or_log_path=or_log_path, parsing_fail_path=parsing_fail_path)
            status = "ok"
            err = ""
        except Exception as e:
            prf = PRF(0.0, 0.0, 0.0, 0, 0, 0)
            status = "error"
            err = repr(e)

        rows.append({
            "group_key": group_key,
            "context": h["context"],
            "pref_type": h["pref_type"],
            "query_turns": h["query_turns"],
            "model_name": h["model_name"],
            "prompt_type": h["prompt_type"],
            "file_name": h["file_name"],
            "json_path": jp,
            "rel_path": rel,
            "status": status,
            "error": err,
            "tp": prf.tp,
            "fp": prf.fp,
            "fn": prf.fn,
            "precision": prf.precision,
            "recall": prf.recall,
            "f1": prf.f1,
            "or_log_path": or_log_path or "",
            "parsing_fail_path": parsing_fail_path or "",
        })

    fieldnames = [
        "group_key",
        "context", "pref_type", "query_turns", "model_name", "prompt_type", "file_name",
        "json_path", "rel_path", "status", "error",
        "tp", "fp", "fn", "precision", "recall", "f1",
        "or_log_path", "parsing_fail_path",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        TP = sum(int(r["tp"]) for r in ok_rows)
        FP = sum(int(r["fp"]) for r in ok_rows)
        FN = sum(int(r["fn"]) for r in ok_rows)
        overall = prf_from_counts(TP, FP, FN)
        print(f"[DONE] files={len(rows)} ok={len(ok_rows)} error={len(rows)-len(ok_rows)}")
        print(f"[AGG over all files' counts] TP={overall.tp} FP={overall.fp} FN={overall.fn}  "
              f"P={overall.precision:.4f} R={overall.recall:.4f} F1={overall.f1:.4f}")
        print(f"[CSV] {out_csv}")
        print(f"[LOGS_DIR] {logs_dir}")
    else:
        print(f"[DONE] files={len(rows)} ok=0 error={len(rows)}")
        print(f"[CSV] {out_csv}")
        print(f"[LOGS_DIR] {logs_dir}")

if __name__ == "__main__":
    main()
