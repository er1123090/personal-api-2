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
from openai import AsyncOpenAI 

# Gemini Library Import
from google import genai
from google.genai import types

# If prompt module is local, keep it; otherwise define dummy variables.
from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE

# Initialize API Keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# [Helper] Load Tools from File
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

# ---------------------------------------------------------
# 1. Load Dataset & Queries (Updated for Multi-turn)
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

# [NEW] Multi-turn Query Data Loader (From Reference Code)
def load_multiturn_query_db(fpath: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Loads the multi-turn query dataset and creates a lookup dictionary.
    Key: (domain, slot, normalized_value_string)
    Value: The full query item (containing 'query' list and 'api_call' list)
    """
    if not os.path.exists(fpath):
        print(f"[Error] Query path not found: {fpath}")
        return {}
    
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    db = {}
    for item in data:
        targets = item.get("target", [])
        for t in targets:
            domain = t.get("domain")
            slot = t.get("slot")
            raw_val = t.get("value")
            
            # Value can be a list ["1", "2"] or a single string "Economy"
            # We index all possible values individually for matching.
            if isinstance(raw_val, list):
                vals_to_index = [str(v) for v in raw_val]
            else:
                vals_to_index = [str(raw_val)]
            
            for val_str in vals_to_index:
                # Key normalization: (Domain, Slot, Value(lowercase))
                key = (domain, slot, val_str.lower())
                db[key] = item
                
    print(f"[Info] Loaded {len(data)} multi-turn queries, indexed into {len(db)} search keys.")
    return db

# [NEW] Helper to format the query list into a string (From Reference Code)
def format_multiturn_conversation(query_list: List[Dict[str, str]]) -> str:
    """
    Converts a list of dicts [{"role":..., "message":...}] into a string.
    """
    lines = []
    for idx, turn in enumerate(query_list):
        role = turn.get("role", "").capitalize()
        message = turn.get("message", "")
        
        if idx == 0 and role == "User":
            # For the first turn, just return the message 
            # (Assumes prompt template adds "User: " or handles it appropriately)
            lines.append(message)
        else:
            lines.append(f"{role}: {message}")
            
    return "\n".join(lines)

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Updated Logic)
# ---------------------------------------------------------
def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_db: Dict[Tuple[str, str, str], Dict[str, Any]], # Changed to DB
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, Any]]: 
    
    results = []

    # Helper function to perform lookup and formatting (From Reference Code)
    def try_add_result(domain, slot, value_str):
        # Key normalization for lookup
        key = (domain, slot, str(value_str).lower())
        
        if key in query_db:
            matched_item = query_db[key]
            
            # 1. Format the multi-turn dialogue query
            query_content_list = matched_item.get("query", [])
            formatted_query = format_multiturn_conversation(query_content_list)
            
            # 2. Extract and Update the Ground Truth (API Call + Target Slot/Value)
            # The goal is to merge the existing api_call with the target slot="value"
            gt_list = matched_item.get("api_call", [])
            base_gt = gt_list[0] if gt_list else f"{domain}()"
            
            # Prepare the argument to add: slot="value"
            # We use value_str because it reflects the matched requirement
            arg_to_add = f'{slot}="{value_str}"'
            
            # Insert logic: Domain(arg1="val") -> Domain(arg1="val", slot="value")
            if base_gt.strip().endswith(")"):
                prefix = base_gt.strip()[:-1]  # Remove trailing ')'
                if prefix.endswith("("):
                    # No existing args: "Service(" -> "Service(slot="val")"
                    new_gt = f'{prefix}{arg_to_add})'
                else:
                    # Existing args: "Service(a="b"" -> "Service(a="b", slot="val")"
                    new_gt = f'{prefix}, {arg_to_add})'
            else:
                # Fallback if format is unexpected
                new_gt = f'{base_gt}, {arg_to_add}'
            
            results.append((formatted_query, new_gt))

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # Parse Domain
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in pref_list: continue

                # Parse Args to find pref slots
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                for slot, value in matches:
                    if slot in target_pref_slots:
                        try_add_result(domain, slot, value)

    # [CASE 2] medium
    elif pref_type == "medium":
        # Extract existing domains to avoid redundancy if needed
        api_calls = example.get("api_calls", [])
        easy_domain_list=[]
        if isinstance(api_calls, list):
            for call_str in api_calls:
                easy_domain_list.append(call_str.split("(")[0].strip() if "(" in call_str else call_str.strip())

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            if pref.get("value_group") in pref_group_data:
                for evidence in pref.get("evidence", []):
                    domain = evidence.get("domain")
                    if domain not in easy_domain_list and domain: 
                        try_add_result(domain, evidence["slot"], evidence["value"])

    # [CASE 3] hard
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data: continue
            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}

            for rule in pref_group_data[current_group_name].get("rules", []):
                candidate_domain = rule.get("domain")
                if candidate_domain and (candidate_domain not in used_domains):
                    target_value = rule.get("value")
                    # Handle boolean conversion for string matching
                    val_str = "True" if isinstance(target_value, bool) and target_value else "False" if isinstance(target_value, bool) else str(target_value)
                    
                    try_add_result(candidate_domain, rule.get("slot"), val_str)

    return results

# ---------------------------------------------------------
# 3. Helpers for History String Construction
# ---------------------------------------------------------
def get_api_calls_string(example: Dict[str, Any]) -> str:
    sessions = example.get("sessions", [])
    collected_apis = []
    for idx, session in enumerate(sessions, start=1):
        api_calls = session.get("api_call", [])
        if isinstance(api_calls, str) and api_calls: api_calls = [api_calls]
        if isinstance(api_calls, list):
            for call in api_calls:
                collected_apis.append(f"[Session {idx}] {call}")
    return "\n".join(collected_apis)

def get_dialogue_history_string(example: Dict[str, Any]) -> str:
    sessions_str = []
    for idx, instruction_data in enumerate(example.get("sessions", []), start=1):
        lines = [f"[Session {idx}]"]
        for turn in instruction_data.get("dialogue", []):
            role = turn.get("role", "").capitalize()
            content = turn.get("message") or turn.get("content") or ""
            if role and content: lines.append(f"{role}: {content}")
        sessions_str.append("\n".join(lines))
    return "\n\n".join(sessions_str)

# ---------------------------------------------------------
# 4. Build Input Prompt
# ---------------------------------------------------------
def build_input_prompt(
    example: Dict[str, Any], 
    current_user_utterance: str, 
    template: str, 
    context_type: str,
    tools_schema: List[Dict] = None 
) -> str:
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    if context_type == "diag-apilist":
        final_context = f"\n--- Dialogue History ---\n{history_str}\n\n--- Past API Calls ---\n{api_str}\n"
    elif context_type == "apilist-only":
        final_context = api_str
    elif context_type == "diag-only":
        final_context = history_str

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
# 5. Call LLM API (Updated for Reasoning Effort)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None,
    reasoning_effort: str = None 
) -> str:
    
    try:
        # --- GEMINI LOGIC (NEW SDK: google-genai) ---
        if "gemini" in model_name.lower():
            if not google_api_key: return "API_KEY_MISSING_GOOGLE"
            
            client = genai.Client(api_key=google_api_key)
            
            config_params = {
                "temperature": 0.0,
            }

            if reasoning_effort and ("gemini-3" in model_name.lower() or "flash" in model_name.lower()):
                config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower() 
                )

            conf = types.GenerateContentConfig(**config_params)
            
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=conf
            )
            
            return response.text.strip()

        # --- OPENAI LOGIC ---
        else:
            if not openai_client: return "API_KEY_MISSING_OPENAI"
            
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "tools": tools_schema if tools_schema else None,
                "tool_choice": "auto" if tools_schema else None,
            }

            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5", "gpt-5.1", "gpt-5-mini"])
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
    utterance: str, 
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
        # NOTE: utterance here is now the full multi-turn formatted string
        current_ex["user_utterance"] = utterance
        current_ex["reference_ground_truth"] = ground_truth
        current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
        current_ex["model_name"] = model_name
        
        prompt = build_input_prompt(
            current_ex, 
            current_user_utterance=utterance, 
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
            "injected_utterance": utterance, # Logs the multi-turn dialogue
            "reasoning_effort": reasoning_effort,
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
    multiturn_query_path: str, # [MODIFIED] Replaced query_map_path
    pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None 
):
    df = load_chains_dataset(input_path)
    
    # [MODIFIED] Use the new Multi-turn Loader
    query_db = load_multiturn_query_db(multiturn_query_path)
    
    tools_schema = load_tools_from_file(tools_schema_path)
    
    if not tools_schema and "gemini" not in model_name.lower():
        print("Warning: Tool schema is empty or missing.")

    openai_client = None
    if openai_api_key:
        openai_client = AsyncOpenAI(api_key=openai_api_key)

    skipped_count = 0
    print(f"Starting ASYNC process... (Model: {model_name}) | Reasoning: {reasoning_effort}")

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

        # [MODIFIED] Pass query_db instead of query_map
        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_db, pref_type, pref_group_path)
        
        if not pairs_list:
            skipped_count += 1; continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            prepared_items.append({
                "original_ex": original_ex,
                "utterance": utterance,
                "ground_truth": ground_truth,
                "sub_idx": sub_idx
            })

    total_tasks = len(prepared_items)
    print(f"Total tasks: {total_tasks}. Skipped: {skipped_count}")

    pbar = tqdm.tqdm(total=total_tasks, desc="Processing Async")

    for item in prepared_items:
        task = asyncio.create_task(
            process_single_item(
                original_ex=item['original_ex'],
                utterance=item['utterance'],
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
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/data/dev_5.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    
    # [MODIFIED] Changed from query_path to multiturn_query_path
    parser.add_argument("--multiturn_query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/query_multiturn.json")
    
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_group.json")
    
    # Tools Schema Argument
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/schema_easy.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group", "imp-pref"], default="imp-zs")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    
    # Reasoning Effort Argument
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None, help="Set reasoning effort for supported models (o1, gpt-5, gemini-3)")

    args = parser.parse_args()

    # Template selection
    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    else: selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            multiturn_query_path=args.multiturn_query_path, # [MODIFIED]
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
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