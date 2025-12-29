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
from google import genai
from google.genai import types

# If prompt module is local, keep it; otherwise define dummy variables.
try:
    from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE
except ImportError:
    IMPLICIT_ZS_PROMPT_TEMPLATE = "Dialogue History:\n{dialogue_history}\n\nUser: {user_utterance}\n\nSchema:\n{preference_schema}"

# Initialize API Keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")


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

# [NEW] Multi-turn Query Loader & Indexer
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
        # Each item has a "target" list, usually one target slot per query context
        for target in item.get("target", []):
            domain = target.get("domain")
            slot = target.get("slot")
            if domain and slot:
                pool[(domain, slot)].append(item)
    
    print(f"[Info] Loaded multi-turn pool with {len(data)} queries, indexed into {len(pool)} domain-slot pairs.")
    return pool

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Updated for Multi-turn)
# ---------------------------------------------------------
def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_map: Dict[str, str], # Single turn map (not used much here but kept for compat)
    pref_type: str, 
    multiturn_pool: Dict[Tuple[str, str], List[Dict]], # [NEW]
    pref_group_path: str = None
) -> List[Tuple[Dict[str, Any], Any]]: 
    """
    Returns a list of tuples: (Input_Context_Dict, Ground_Truth_List)
    Input_Context_Dict keys: 'history' (str), 'last_utterance' (str), 'base_api_call' (str)
    """
    
    results = []

    # Helper function to ensure values are strings
    def to_str(val):
        if isinstance(val, bool):
            return "True" if val else "False"
        return str(val)

    # 1. Identify the Target Preference (Domain, Slot, Values) based on type
    target_preferences = [] # List of {'domain': str, 'slot': str, 'values': List[str]}

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # Parse domain
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in pref_list: continue

                # Regex to extract key-value pairs
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                ground_truth_pref_slots = pref_list.get(domain, [])
                
                for slot, value in matches:
                    if slot in ground_truth_pref_slots:
                        target_preferences.append({
                            "domain": domain,
                            "slot": slot,
                            "values": [to_str(value)]
                        })

    # [CASE 2] medium
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name in pref_group_data:
                seen_ds = set()
                for evidence in pref.get("evidence", []):
                    domain = evidence.get("domain")
                    slot = evidence.get("slot")
                    # Collect all values in list
                    values_list = [to_str(v.get("value")) for v in evidence.get("values", []) if v.get("value") is not None]
                    
                    if domain and slot and values_list and (domain, slot) not in seen_ds:
                        target_preferences.append({
                            "domain": domain,
                            "slot": slot,
                            "values": values_list
                        })
                        seen_ds.add((domain, slot))

    # [CASE 3] hard
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)
        
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data: continue
            
            rules = pref_group_data[current_group_name].get("rules", [])
            # Group values by domain/slot
            ds_map = defaultdict(list)
            for rule in rules:
                d = rule.get("domain")
                s = rule.get("slot")
                v = rule.get("value")
                if d and s and v is not None:
                    ds_map[(d, s)].append(to_str(v))
            
            for (d, s), v_list in ds_map.items():
                target_preferences.append({
                    "domain": d,
                    "slot": s,
                    "values": list(set(v_list)) # dedup
                })

    # 2. Match with Multi-turn Pool and Construct Result
    for target in target_preferences:
        key = (target["domain"], target["slot"])
        candidates = multiturn_pool.get(key)
        
        if not candidates:
            # print(f"No multi-turn query found for {key}")
            continue
            
        # Strategy: Pick the first match or round-robin if needed. 
        # For consistency with "count must be same as single turn", we map 1 eval instance -> 1 query.
        # We can just pick the first one for simplicity, or random. 
        # Since the provided JSON has unique queries per slot usually, picking index 0 is safe.
        mt_query = candidates[0] 
        
        # 2.1 Format Dialogue History
        dialogue_list = mt_query.get("query", [])
        history_lines = []
        last_user_msg = ""
        
        # Split history and last user turn
        # Assuming the structure is User -> Assistant -> User ...
        # The prompt usually expects the history UP TO the final user turn.
        
        temp_history = []
        for turn in dialogue_list:
            role = turn.get("role", "").capitalize()
            msg = turn.get("message") or turn.get("content") or ""
            temp_history.append((role, msg))
            
        # Pop the last user message as the "current utterance"
        if temp_history and temp_history[-1][0] == "User":
            last_user_msg = temp_history[-1][1]
            prior_turns = temp_history[:-1]
        else:
            # Fallback if no user message at end
            last_user_msg = "Please help me."
            prior_turns = temp_history

        # Build string for history
        history_str = ""
        for role, msg in prior_turns:
            history_str += f"{role}: {msg}\n"
            
        # 2.2 Construct Base API Call (Context)
        # The multi-turn JSON has "api_call" which is the 'partial' or 'base' call derived from context
        base_api_calls = mt_query.get("api_call", [])
        base_api_str = base_api_calls[0] if base_api_calls else ""
        
        # 2.3 Construct Ground Truth
        # GT = Base API Call Context + Injected Preference Value
        # We return the structured object, but note the 'value' is from the Evaluation Instance (target['values'])
        # The 'base_api_call' context is stored in the input wrapper for verification if needed.
        
        gt_obj = [{
            "domain": target["domain"],
            "slot": target["slot"],
            "value": target["values"], # The implicit preference to learn/inject
            "base_api_call": base_api_str # Context from history
        }]
        
        input_context = {
            "history": history_str.strip(),
            "last_utterance": last_user_msg,
            "base_api_call": base_api_str
        }
        
        results.append((input_context, gt_obj))

    return results

# ---------------------------------------------------------
# 4. Build Input Prompt (Updated)
# ---------------------------------------------------------
def build_input_prompt(
    example: Dict[str, Any], 
    input_context: Dict[str, Any], # Changed from string to dict
    template: str, 
    context_type: str,
    tools_schema: List[Dict] = None 
) -> str:
    
    history_str = input_context.get("history", "")
    current_user_utterance = input_context.get("last_utterance", "")
    
    # [Note] The context_type logic (diag-apilist etc) can be adapted.
    # For multi-turn, we primarily use the history string generated from the JSON.
    
    final_context = ""
    if "diag" in context_type:
        final_context = f"\n--- Dialogue History ---\n{history_str}\n"
    
    # If the user wants to see past API calls (if any exist in the multi-turn json sessions), 
    # we could add them, but the provided JSON structure puts context api_call in a separate field.
    # We will stick to the dialogue history as the main context source.

    schema_str = ""
    if tools_schema:
        schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False)
    else:
        schema_str = "No specific schema provided."

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
# 5. Call LLM API (Async)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None,
    reasoning_effort: str = None
) -> str:
    try:
        # --- GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key: return "API_KEY_MISSING_GOOGLE"
            client = genai.Client(api_key=google_api_key)
            config_params = {"temperature": 0.0}

            if reasoning_effort and ("gemini-3" in model_name.lower() or "flash" in model_name.lower()):
                config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower()
                )

            conf = types.GenerateContentConfig(**config_params)
            response = await client.aio.models.generate_content(
                model=model_name, contents=prompt, config=conf
            )
            return response.text.strip()

        # --- OPENAI LOGIC ---
        else:
            if not openai_client: return "API_KEY_MISSING_OPENAI"
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools_schema if tools_schema else None,
                "tool_choice": "auto" if tools_schema else None,
            }
            
            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5"])
            if is_reasoning_model and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower()

            response = await openai_client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            raw_content = message.content or ""

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                    args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                    output = f"{func_name}({', '.join(args_str_list)})"
                except:
                    output = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
            else:
                # Remove <think> tags for DeepSeek compatibility if needed
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
                if match:
                    output = match.group(0).strip()
                else:
                    output = clean_content
            
            return output

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        return f"API_ERROR: {str(e)}"

# ---------------------------------------------------------
# 6. Pipeline (Async Version)
# ---------------------------------------------------------
async def process_single_item(
    original_ex: Dict[str, Any],
    input_context: Dict[str, Any], 
    ground_truth: Any, 
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
    pbar: tqdm.tqdm,
    reasoning_effort: str = None
):
    async with semaphore:
        current_ex = copy.deepcopy(original_ex)
        # Store metadata
        current_ex["user_utterance"] = input_context["last_utterance"]
        current_ex["dialogue_history"] = input_context["history"]
        current_ex["base_api_call"] = input_context["base_api_call"]
        current_ex["reference_ground_truth"] = ground_truth
        current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
        current_ex["model_name"] = model_name
        
        prompt = build_input_prompt(
            current_ex, 
            input_context=input_context, 
            template=prompt_template, 
            context_type=context_type,
            tools_schema=tools_schema 
        )

        llm_output = await call_llm_api_async(
            prompt, model_name, openai_client, tools_schema, reasoning_effort
        )

        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": current_ex["example_id"],
            "example_id_sub": current_ex["example_id_sub"], 
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "is_multiturn": True,
            "dialogue_history": input_context["history"],
            "user_utterance": input_context["last_utterance"],
            "base_api_call_context": input_context["base_api_call"],
            "reference_ground_truth": ground_truth, 
            "model_input": prompt,
            "model_output": llm_output,
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        current_ex["llm_output"] = llm_output
        pbar.update(1)
        return current_ex

async def process_with_llm_async(
    input_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    multiturn_query_path: str, # [NEW]
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    
    # [NEW] Load Multi-turn Pool
    multiturn_pool = load_multiturn_pool(multiturn_query_path)
    if not multiturn_pool:
        print("CRITICAL: Multi-turn pool is empty. Exiting.")
        return

    openai_client = None
    if openai_api_key:
        openai_client = AsyncOpenAI(api_key=openai_api_key)

    skipped_count = 0
    print(f"Starting ASYNC process (Multi-turn)... (Model: {model_name})")

    tasks = []
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    
    print("Preparing tasks...")
    prepared_items = []
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()

        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        # [NEW] Pass multiturn_pool
        pairs_list = assign_user_utterances(
            pref_list_path, original_ex, query_map, pref_type, 
            multiturn_pool=multiturn_pool, 
            pref_group_path=pref_group_path
        )
        
        if not pairs_list:
            skipped_count += 1; continue

        for sub_idx, (input_context, ground_truth) in enumerate(pairs_list):
            prepared_items.append({
                "original_ex": original_ex,
                "input_context": input_context,
                "ground_truth": ground_truth,
                "sub_idx": sub_idx
            })

    total_tasks = len(prepared_items)
    print(f"Total tasks: {total_tasks}. Skipped due to no match/pref: {skipped_count}")

    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    for item in prepared_items:
        task = asyncio.create_task(
            process_single_item(
                original_ex=item['original_ex'],
                input_context=item['input_context'],
                ground_truth=item['ground_truth'],
                sub_idx=item['sub_idx'],
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
                pbar=pbar,
                reasoning_effort=reasoning_effort
            )
        )
        tasks.append(task)
    
    processed_data = await asyncio.gather(*tasks)
    pbar.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/data/dev_6.json")
    parser.add_argument("--output_path", type=str, default="output_multiturn.json")
    parser.add_argument("--log_path", type=str, default="process_multiturn.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/query_singleturn.json")
    
    # [NEW] Multi-turn query path argument
    parser.add_argument("--multiturn_query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/query_multiturn.json", help="Path to the multiturn query json file")
    
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/schema_easy.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, default="imp-zs")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--reasoning_effort", type=str, default=None)

    args = parser.parse_args()

    # Template selection
    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    else: selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE # Default fallback

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            multiturn_query_path=args.multiturn_query_path, # Passed here
            tools_schema_path=args.tools_schema_path,
            prompt_template=selected_template,
            prompt_type_name=args.prompt_type,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort
        )
    )