import tqdm
import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import re
import copy
import asyncio
from collections import defaultdict
from openai import AsyncOpenAI

# ---------------------------------------------------------
# Prompt import (keep as-is)
# ---------------------------------------------------------
try:
    from prompt import (
        IMPLICIT_ZS_PROMPT_TEMPLATE,
        IMPLICIT_FS_PROMPT_TEMPLATE,
        IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE,
    )
except ImportError:
    IMPLICIT_ZS_PROMPT_TEMPLATE = "Dialogue History:\n{dialogue_history}\n\nUser: {user_utterance}\n\nSchema:\n{preference_schema}"
    IMPLICIT_FS_PROMPT_TEMPLATE = IMPLICIT_ZS_PROMPT_TEMPLATE
    IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = IMPLICIT_ZS_PROMPT_TEMPLATE

# ---------------------------------------------------------
# [Helper] Load Tools & Datasets
# ---------------------------------------------------------
def load_tools_from_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"[Warning] Tools schema file not found at {file_path}")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        print(f"[Info] Successfully loaded {len(tools)} tools from {file_path}")
        return tools
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pref_group(fpath: str) -> Dict[str, Any]:
    if not fpath or not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# [Helper] Multi-turn Query Loader & Indexer (SAME as above)
# ---------------------------------------------------------
def load_multiturn_pool(fpath: str) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Loads the multi-turn query dataset and indexes it by (domain, slot).
    Returns: Dict[(domain, slot), List[query_objects]]
    """
    if not os.path.exists(fpath):
        print(f"[Error] Multi-turn query file not found: {fpath}")
        return {}

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    pool = defaultdict(list)
    for item in data:
        for target in item.get("target", []):
            domain = target.get("domain")
            slot = target.get("slot")
            if domain and slot:
                pool[(domain, slot)].append(item)

    print(f"[Info] Loaded multi-turn pool with {len(data)} queries, indexed into {len(pool)} domain-slot pairs.")
    return pool

# ---------------------------------------------------------
# [Helper] pref_group -> value candidates map (SAME as above)
# ---------------------------------------------------------
def _to_str(val: Any) -> str:
    if isinstance(val, bool):
        return "True" if val else "False"
    return str(val)

def build_group_value_map(pref_group_data: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, str], List[str]]]:
    """
    group_name -> {(domain, slot): [values...]}
    """
    group_map: Dict[str, Dict[Tuple[str, str], List[str]]] = {}
    for gname, gobj in (pref_group_data or {}).items():
        ds_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for rule in gobj.get("rules", []):
            d = rule.get("domain")
            s = rule.get("slot")
            v = rule.get("value")
            if not d or not s or v is None:
                continue
            vv = _to_str(v)
            if vv not in ds_map[(d, s)]:
                ds_map[(d, s)].append(vv)
        group_map[gname] = dict(ds_map)
    return group_map

# ---------------------------------------------------------
# [Helper] Parse + Render + Merge base_api_call with injected slot/value
# ---------------------------------------------------------
def _parse_func_call(call_str: str) -> Tuple[str, Dict[str, str]]:
    """
    "GetEvents(number_of_tickets="3", foo="bar")" -> ("GetEvents", {"number_of_tickets":"3","foo":"bar"})
    """
    call_str = (call_str or "").strip()
    if not call_str:
        return "", {}

    if "(" not in call_str or ")" not in call_str:
        return call_str, {}

    func = call_str.split("(", 1)[0].strip()
    args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]

    pattern = r'(\w+)=["\']([^"\']+)["\']'
    matches = re.findall(pattern, args_content)
    args = {k: str(v) for k, v in matches}
    return func, args

def _render_func_call(func: str, args: Dict[str, str]) -> str:
    func = (func or "").strip()
    if not func:
        return ""
    if not args:
        return f"{func}()"
    parts = [f'{k}="{args[k]}"' for k in sorted(args.keys())]
    return f"{func}({', '.join(parts)})"

def build_reference_ground_truth(base_api_call: str, func_domain: str, slot: str, values: List[str]) -> List[str]:
    """
    Merge base_api_call args with injected (slot -> each value).
    Return List[str] (multiple GT strings possible).
    """
    base_api_call = (base_api_call or "").strip()
    func_domain = (func_domain or "").strip()
    slot = (slot or "").strip()

    base_func, base_args = _parse_func_call(base_api_call) if base_api_call else ("", {})
    func = base_func if base_func else func_domain
    if not func:
        return []

    if values is None:
        values_list: List[str] = []
    elif isinstance(values, list):
        values_list = [str(v).strip() for v in values if str(v).strip()]
    else:
        values_list = [str(values).strip()] if str(values).strip() else []

    if not slot or not values_list:
        return [_render_func_call(func, base_args)]

    gts = []
    for v in values_list:
        merged = dict(base_args)
        merged[slot] = v
        gts.append(_render_func_call(func, merged))
    return gts

# ---------------------------------------------------------
# (A) NEW assign_user_utterances for vLLM script:
#     - query generation: use multiturn_pool
#     - gt generation: base_api_call + pref_group candidate values
#     - output format: (input_context_dict, reference_ground_truth_list)
# ---------------------------------------------------------
def assign_user_utterances_multiturn(
    pref_list_path: str,
    example: Dict[str, Any],
    query_map: Dict[str, str],  # kept for compatibility
    pref_type: str,
    multiturn_pool: Dict[Tuple[str, str], List[Dict]],
    pref_group_path: str = None
) -> List[Tuple[Dict[str, Any], List[str]]]:

    results: List[Tuple[Dict[str, Any], List[str]]] = []

    pref_group_data = load_pref_group(pref_group_path) if pref_group_path else {}
    group_value_map = build_group_value_map(pref_group_data) if pref_group_data else {}

    target_preferences: List[Dict[str, Any]] = []

    # -------------------------
    # EASY: value from api_calls (single exact value)
    # -------------------------
    if pref_type == "easy":
        if not os.path.exists(pref_list_path):
            return []
        with open(pref_list_path, "r", encoding="utf-8") as f:
            pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try:
                        args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError:
                        continue
                else:
                    domain = call_str.strip()
                    args_content = ""

                if domain not in pref_list:
                    continue

                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                pref_slots = pref_list.get(domain, [])

                for slot, value in matches:
                    if slot in pref_slots:
                        target_preferences.append({
                            "domain": domain,
                            "slot": slot,
                            "values": [_to_str(value)]
                        })

    # -------------------------
    # MEDIUM: evidence gives (domain,slot) but candidate values come from pref_group rules
    # -------------------------
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        if not group_value_map:
            return []

        for pref in prefs:
            group_name = pref.get("value_group")
            if not group_name or group_name not in group_value_map:
                continue

            ds_values = group_value_map[group_name]
            seen_ds = set()

            for evidence in pref.get("evidence", []):
                d = evidence.get("domain")
                s = evidence.get("slot")
                if not d or not s or (d, s) in seen_ds:
                    continue

                cand_vals = ds_values.get((d, s), [])
                if cand_vals:
                    target_preferences.append({
                        "domain": d,
                        "slot": s,
                        "values": cand_vals  # ALL values from rules
                    })
                    seen_ds.add((d, s))

    # -------------------------
    # HARD: for group_name, evaluate ALL (domain,slot) in rules (all values)
    # -------------------------
    elif pref_type == "hard":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs:
            return []
        if not group_value_map:
            return []

        for pref in prefs:
            group_name = pref.get("value_group")
            if not group_name or group_name not in group_value_map:
                continue

            ds_values = group_value_map[group_name]
            for (d, s), cand_vals in ds_values.items():
                if cand_vals:
                    target_preferences.append({
                        "domain": d,
                        "slot": s,
                        "values": cand_vals
                    })
    else:
        return []

    # -------------------------
    # Match multiturn_pool and build (history,last_utterance,base_api_call) + GT strings
    # -------------------------
    for target in target_preferences:
        key = (target["domain"], target["slot"])
        candidates = multiturn_pool.get(key)
        if not candidates:
            continue

        mt_query = candidates[0]

        dialogue_list = mt_query.get("query", [])
        temp_history = []
        for turn in dialogue_list:
            role = turn.get("role", "").capitalize()
            msg = turn.get("message") or turn.get("content") or ""
            temp_history.append((role, msg))

        if temp_history and temp_history[-1][0] == "User":
            last_user_msg = temp_history[-1][1]
            prior_turns = temp_history[:-1]
        else:
            last_user_msg = "Please help me."
            prior_turns = temp_history

        history_str = ""
        for role, msg in prior_turns:
            history_str += f"{role}: {msg}\n"

        base_api_calls = mt_query.get("api_call", [])
        base_api_str = base_api_calls[0] if base_api_calls else ""

        reference_ground_truth = build_reference_ground_truth(
            base_api_call=base_api_str,
            func_domain=target["domain"],
            slot=target["slot"],
            values=target["values"]
        )

        input_context = {
            "history": history_str.strip(),
            "last_utterance": last_user_msg,
            "base_api_call": base_api_str
        }

        results.append((input_context, reference_ground_truth))

    return results

# ---------------------------------------------------------
# Prompt building for multiturn (match "above code" style)
# ---------------------------------------------------------
def build_input_prompt_multiturn(
    input_context: Dict[str, Any],
    template: str,
    context_type: str,
    tools_schema: List[Dict] = None
) -> str:
    history_str = input_context.get("history", "")
    current_user_utterance = input_context.get("last_utterance", "")

    final_context = ""
    if "diag" in context_type:
        final_context = f"\n--- Dialogue History ---\n{history_str}\n"

    schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False) if tools_schema else "No specific schema provided."

    try:
        prompt = template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
            preference_schema=schema_str
        )
    except KeyError:
        prompt = template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip()
        )

    return prompt

# ---------------------------------------------------------
# vLLM API Call + parsing (keep your hybrid logic)
# ---------------------------------------------------------
async def vllm_call_and_parse(
    client: AsyncOpenAI,
    prompt: str,
    tools_schema: List[Dict],
    model_name: str,
) -> str:
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            tools=tools_schema if tools_schema else None,
            tool_choice="auto" if tools_schema else None,
            temperature=0.0,
        )

        message = response.choices[0].message
        raw_content = message.content or ""

        # 1) tool call
        if getattr(message, "tool_calls", None):
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
                args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                return f"{func_name}({', '.join(args_str_list)})"
            except Exception:
                return f"ERROR_JSON_PARSE: {tool_call.function.arguments}"

        # 2) fallback parse
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
        if match:
            return match.group(0).strip()
        return clean_content

    except Exception as e:
        return f"ERROR_API: {str(e)}"

# ---------------------------------------------------------
# Single item worker (vLLM) - now takes multiturn input_context + List[str] GT
# ---------------------------------------------------------
async def process_single_example_multiturn_vllm(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    log_path: str,
    prompt: str,
    tools_schema: List[Dict],
    model_name: str,
    current_ex: Dict[str, Any],
    prompt_type_name: str,
    context_type: str,
    pref_type: str,
    input_context: Dict[str, Any],
    reference_ground_truth: List[str],
    pbar: tqdm.tqdm
) -> Dict[str, Any]:

    async with semaphore:
        llm_output = await vllm_call_and_parse(
            client=client,
            prompt=prompt,
            tools_schema=tools_schema,
            model_name=model_name
        )

        # log record (match upper style)
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": current_ex.get("example_id", "unknown"),
            "example_id_sub": current_ex.get("example_id_sub", "unknown"),
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "is_multiturn": True,
            "dialogue_history": input_context.get("history", ""),
            "user_utterance": input_context.get("last_utterance", ""),
            "base_api_call_context": input_context.get("base_api_call", ""),
            "reference_ground_truth": reference_ground_truth,
            "model_input": prompt,
            "model_output": llm_output,
        }

        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        pbar.update(1)

        result_ex = copy.deepcopy(current_ex)
        # store same fields as the "above code"
        result_ex["dialogue_history"] = input_context.get("history", "")
        result_ex["user_utterance"] = input_context.get("last_utterance", "")
        result_ex["base_api_call"] = input_context.get("base_api_call", "")
        result_ex["reference_ground_truth"] = reference_ground_truth
        result_ex["llm_output"] = llm_output
        return result_ex

# ---------------------------------------------------------
# vLLM main pipeline (ASYNC) - updated to use multiturn query & GT format
# ---------------------------------------------------------
async def process_with_vllm_server_async(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    multiturn_query_path: str,  # NEW
    pref_list_path: str,
    pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str,
    prompt_type_name: str,
    context_type: str,
    pref_type: str,
    model_name: str,
    vllm_url: str,
    concurrency: int
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)

    # tools schema
    print(f"Loading tools schema from {tools_schema_path}...")
    tools_schema = load_tools_from_file(tools_schema_path)
    if not tools_schema:
        print("[Warning] Tools schema is empty! 'tool_choice' will be disabled.")

    # multiturn pool (NEW)
    multiturn_pool = load_multiturn_pool(multiturn_query_path)
    if not multiturn_pool:
        print("CRITICAL: Multi-turn pool is empty. Exiting.")
        return

    # vLLM client
    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    print(f"Connected to vLLM Server at {vllm_url} (Max Concurrency: {concurrency})")
    print(f"Starting process... (Model: {model_name})")

    pending_items = []
    skipped_count = 0

    print("Preparing tasks...")

    for _, row in df.iterrows():
        original_ex = row.to_dict()

        if pref_type == "easy":
            if not original_ex.get("api_calls"):
                skipped_count += 1
                continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"):
                skipped_count += 1
                continue

        # ✅ NEW: multiturn-based query + GT generation
        pairs_list = assign_user_utterances_multiturn(
            pref_list_path=pref_list_path,
            example=original_ex,
            query_map=query_map,
            pref_type=pref_type,
            multiturn_pool=multiturn_pool,
            pref_group_path=pref_group_path
        )

        if not pairs_list:
            skipped_count += 1
            continue

        for sub_idx, (input_context, reference_ground_truth) in enumerate(pairs_list):
            current_ex = copy.deepcopy(original_ex)
            current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
            current_ex["model_name"] = model_name

            prompt = build_input_prompt_multiturn(
                input_context=input_context,
                template=prompt_template,
                context_type=context_type,
                tools_schema=tools_schema
            )

            pending_items.append({
                "prompt": prompt,
                "current_ex": current_ex,
                "input_context": input_context,
                "reference_ground_truth": reference_ground_truth
            })

    total_tasks = len(pending_items)
    print(f"Total tasks prepared: {total_tasks} (Skipped source examples: {skipped_count})")

    pbar = tqdm.tqdm(total=total_tasks, desc="Async Inference (vLLM)")

    tasks = []
    for item in pending_items:
        tasks.append(
            process_single_example_multiturn_vllm(
                client=client,
                semaphore=semaphore,
                file_lock=file_lock,
                log_path=log_path,
                prompt=item["prompt"],
                tools_schema=tools_schema,
                model_name=model_name,
                current_ex=item["current_ex"],
                prompt_type_name=prompt_type_name,
                context_type=context_type,
                pref_type=pref_type,
                input_context=item["input_context"],
                reference_ground_truth=item["reference_ground_truth"],
                pbar=pbar
            )
        )

    processed_data = await asyncio.gather(*tasks)
    pbar.close()
    await client.close()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Saved -> {output_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output_multiturn_vllm.json")
    parser.add_argument("--log_path", type=str, default="process_multiturn_vllm.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json")

    # ✅ NEW: multiturn query pool
    parser.add_argument("--multiturn_query_path", type=str, default="/data/minseo/experiments4/query_multiturn.json")

    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/tools_schema.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group"], default="imp-zs")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=50)

    args = parser.parse_args()

    if args.prompt_type == "imp-zs":
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs":
        selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group":
        selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else:
        selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_with_vllm_server_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            multiturn_query_path=args.multiturn_query_path,  # ✅ NEW
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            tools_schema_path=args.tools_schema_path,
            prompt_template=selected_template,
            prompt_type_name=args.prompt_type,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            vllm_url=args.vllm_url,
            concurrency=args.concurrency
        )
    )
