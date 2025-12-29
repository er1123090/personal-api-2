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

# Gemini Library Import
import google.generativeai as genai

# If prompt module is local, keep it; otherwise define dummy variables.
try:
    from prompt import (
        IMPLICIT_ZS_PROMPT_TEMPLATE,
        IMPLICIT_FS_PROMPT_TEMPLATE,
        IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE,
    )
except ImportError:
    IMPLICIT_ZS_PROMPT_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"
    IMPLICIT_FS_PROMPT_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"
    IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"

# Initialize API Keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)

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
# [NEW] Multi-turn Query Loader & Indexer  (same as above code)
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
# [Helper] pref_group -> value candidates map (same as above code)
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
    Parse "GetEvents(number_of_tickets="3", foo="bar")" -> ("GetEvents", {"number_of_tickets":"3","foo":"bar"})
    If no parentheses: "GetEvents" -> ("GetEvents", {})
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
    """
    Canonical render: Func(k="v", ...) with keys sorted for determinism.
    """
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
# ✅ NEW: multiturn assigner (same behavior as "above code")
#   - returns: List[(input_context_dict, reference_ground_truth_list)]
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

    # [CASE 1] easy: exact values from api_calls
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

    # [CASE 2] medium: evidence gives (domain,slot), values come from pref_group rules (ALL)
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

            ds_values = group_value_map[group_name]  # {(domain,slot): [values...]}
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
                        "values": cand_vals
                    })
                    seen_ds.add((d, s))

    # [CASE 3] hard: for group_name, take ALL (domain,slot)->ALL values in rules
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

    # match multiturn_pool and create (history,last_utterance,base_api_call) + multiple GT strings
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
# ✅ NEW: prompt building for multiturn (same as above code’s style)
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
# 5. Call LLM API (KEEP reasoning extraction)
#   - BUT: also support vLLM via openai_client.base_url (your call site decides)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str,
    model_name: str,
    openai_client: AsyncOpenAI = None,
    tools_schema: List[Dict] = None,
    baseline_prompt_path: str = "/data/minseo/personal-tool/conv_api/experiments3/new_baseline_prompt_update.txt"
) -> Dict[str, Any]:
    """
    Returns:
    {
        "output": str,
        "reasoning": Optional[str],
        "error": Optional[str]
    }
    """
    result = {"output": "", "reasoning": None, "error": None}

    try:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()
    except FileNotFoundError:
        baseline_prompt = "You are a helpful assistant."

    try:
        # --- GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key:
                result["error"] = "API_KEY_MISSING_GOOGLE"
                return result

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=baseline_prompt
            )

            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )

            full_text = ""
            reasoning_text = ""

            try:
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            reasoning_text += str(part.thought) + "\n"
                        elif hasattr(part, 'text'):
                            full_text += part.text

                    if not full_text and not reasoning_text:
                        full_text = response.text
                else:
                    full_text = ""
            except Exception as parse_e:
                full_text = response.text if hasattr(response, 'text') else str(response)
                print(f"[Warning] Gemini parsing issue: {parse_e}")

            result["output"] = (full_text or "").strip()
            if reasoning_text.strip():
                result["reasoning"] = reasoning_text.strip()
            return result

        # --- OPENAI / vLLM (OpenAI-compatible) LOGIC ---
        else:
            if not openai_client:
                result["error"] = "API_KEY_MISSING_OPENAI"
                return result

            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                temperature=0.0,
            )

            message = response.choices[0].message
            raw_content = message.content or ""

            # Tool call parsing
            if getattr(message, "tool_calls", None):
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                    args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                    result["output"] = f"{func_name}({', '.join(args_str_list)})"
                    return result
                except Exception:
                    result["output"] = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
                    return result

            # Text parsing (+ reasoning extraction via <think>)
            think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
            if think_match:
                result["reasoning"] = think_match.group(1).strip()

            clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
            result["output"] = match.group(0).strip() if match else clean_content
            return result

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        result["error"] = f"API_ERROR: {str(e)}"
        return result

# ---------------------------------------------------------
# ✅ MODIFIED: process_single_item for multiturn input_context + List[str] GT
#     - keep reasoning 저장 기능 유지
# ---------------------------------------------------------
async def process_single_item_multiturn(
    original_ex: Dict[str, Any],
    input_context: Dict[str, Any],
    reference_ground_truth: List[str],
    sub_idx: int,
    model_name: str,
    prompt_template: str,
    context_type: str,
    prompt_type_name: str,
    pref_type: str,
    openai_client: AsyncOpenAI,
    tools_schema: List[Dict],
    log_path: str,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    pbar: tqdm.tqdm
) -> Dict[str, Any]:
    async with semaphore:
        current_ex = copy.deepcopy(original_ex)

        # ✅ align with "above code" fields
        current_ex["dialogue_history"] = input_context.get("history", "")
        current_ex["user_utterance"] = input_context.get("last_utterance", "")
        current_ex["base_api_call"] = input_context.get("base_api_call", "")
        current_ex["reference_ground_truth"] = reference_ground_truth
        current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
        current_ex["model_name"] = model_name

        # ✅ use multiturn prompt builder
        prompt = build_input_prompt_multiturn(
            input_context=input_context,
            template=prompt_template,
            context_type=context_type,
            tools_schema=tools_schema
        )

        llm_result = await call_llm_api_async(
            prompt=prompt,
            model_name=model_name,
            openai_client=openai_client,
            tools_schema=tools_schema
        )

        llm_output = llm_result.get("output", "")
        llm_reasoning = llm_result.get("reasoning", None)
        error_msg = llm_result.get("error", None)
        if error_msg:
            llm_output = error_msg

        # ✅ log format aligned with above + keep reasoning
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": current_ex.get("example_id", "unknown"),
            "example_id_sub": current_ex["example_id_sub"],
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "is_multiturn": True,
            "dialogue_history": input_context.get("history", ""),
            "user_utterance": input_context.get("last_utterance", ""),
            "base_api_call_context": input_context.get("base_api_call", ""),
            "reference_ground_truth": reference_ground_truth,  # ✅ List[str]
            "model_input": prompt,
            "model_output": llm_output,
            "model_reasoning": llm_reasoning,  # ✅ keep reasoning
        }

        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        current_ex["llm_output"] = llm_output
        current_ex["llm_reasoning"] = llm_reasoning

        pbar.update(1)
        return current_ex

# ---------------------------------------------------------
# ✅ MODIFIED: main async pipeline
#   - uses multiturn_pool
#   - assigns input_context + List[str] GT
#   - keeps reasoning saved
#   - supports BOTH OpenAI and vLLM by choosing client base_url
# ---------------------------------------------------------
async def process_with_llm_async(
    input_path: str,
    output_path: str,
    log_path: str,
    query_map_path: str,
    pref_list_path: str,
    pref_group_path: str,
    multiturn_query_path: str,   # ✅ NEW
    tools_schema_path: str,
    prompt_template: str,
    prompt_type_name: str,
    context_type: str,
    pref_type: str,
    model_name: str,
    concurrency: int = 10,
    vllm_url: Optional[str] = None  # ✅ NEW: if set, use vLLM endpoint
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)

    tools_schema = load_tools_from_file(tools_schema_path)
    if not tools_schema and "gemini" not in model_name.lower():
        print("[Warning] Tool schema is empty or missing.")

    multiturn_pool = load_multiturn_pool(multiturn_query_path)
    if not multiturn_pool:
        print("CRITICAL: Multi-turn pool is empty. Exiting.")
        return

    # client init: OpenAI or vLLM (OpenAI-compatible)
    openai_client = None
    if "gemini" not in model_name.lower():
        if vllm_url:
            # vLLM: OpenAI-compatible server
            openai_client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
        else:
            if openai_api_key:
                openai_client = AsyncOpenAI(api_key=openai_api_key)

    skipped_count = 0
    print(f"Starting ASYNC process... (Model: {model_name})")
    if vllm_url:
        print(f"Using vLLM endpoint: {vllm_url}")

    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    prepared_items = []
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

        # ✅ multiturn query + GT(list) generation
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
            prepared_items.append({
                "original_ex": original_ex,
                "input_context": input_context,
                "reference_ground_truth": reference_ground_truth,
                "sub_idx": sub_idx
            })

    total_tasks = len(prepared_items)
    print(f"Total tasks: {total_tasks}. Skipped: {skipped_count}")

    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    tasks = []
    for item in prepared_items:
        tasks.append(
            asyncio.create_task(
                process_single_item_multiturn(
                    original_ex=item["original_ex"],
                    input_context=item["input_context"],
                    reference_ground_truth=item["reference_ground_truth"],
                    sub_idx=item["sub_idx"],
                    model_name=model_name,
                    prompt_template=prompt_template,
                    context_type=context_type,
                    prompt_type_name=prompt_type_name,
                    pref_type=pref_type,
                    openai_client=openai_client,
                    tools_schema=tools_schema,
                    log_path=log_path,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    pbar=pbar
                )
            )
        )

    processed_data = await asyncio.gather(*tasks)
    pbar.close()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Saved -> {output_path}")

    # close vLLM client explicitly
    try:
        if vllm_url and openai_client:
            await openai_client.close()
    except Exception:
        pass

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output_multiturn.json")
    parser.add_argument("--log_path", type=str, default="process_multiturn.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json")

    parser.add_argument("--multiturn_query_path", type=str, required=True, help="Path to multiturn query json (indexed by domain-slot)")

    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_group.json")

    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/tools_schema.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group"], default="imp-zs")

    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)

    # ✅ OPTIONAL: if set, OpenAI calls go to vLLM server
    parser.add_argument("--vllm_url", type=str, default=None, help="e.g., http://localhost:8000/v1")

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
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            multiturn_query_path=args.multiturn_query_path,   # ✅ NEW
            tools_schema_path=args.tools_schema_path,
            prompt_template=selected_template,
            prompt_type_name=args.prompt_type,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            concurrency=args.concurrency,
            vllm_url=args.vllm_url,  # ✅ NEW
        )
    )
