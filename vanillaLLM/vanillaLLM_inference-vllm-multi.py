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
import itertools 
from openai import AsyncOpenAI 

# ---------------------------------------------------------
# [Helper] Load Tools & Data
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

def load_multiturn_data(fpath: str) -> Dict[str, Any]:
    """
    Loads the multi-turn query JSON file.
    Structure: { "GetRestaurants": [ { "query": [...], "api_call": [...] } ] }
    """
    if not os.path.exists(fpath):
        print(f"[Error] Multi-turn query file not found: {fpath}")
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# [Helper] API String Manipulation (Multi-turn Logic)
# ---------------------------------------------------------
def parse_api_call_to_dict(api_str: str) -> Tuple[str, Dict[str, str]]:
    """
    Parses 'GetRestaurants(city="Vallejo", date="2019-03-05")' 
    into ('GetRestaurants', {'city': 'Vallejo', 'date': '2019-03-05'})
    """
    if "(" not in api_str:
        return api_str.strip(), {}
    
    domain = api_str.split("(")[0].strip()
    try:
        args_content = api_str.split("(", 1)[1].rsplit(")", 1)[0]
    except IndexError:
        return domain, {}

    # Regex to capture key="value" or key='value'
    pattern = r'(\w+)=["\']([^"\']+)["\']'
    matches = re.findall(pattern, args_content)
    
    args_dict = {k: v for k, v in matches}
    return domain, args_dict

def generate_single_api_string(domain: str, args_dict: Dict[str, str]) -> str:
    sorted_keys = sorted(args_dict.keys())
    args_parts = [f'{key}="{args_dict[key]}"' for key in sorted_keys]
    return f"{domain}({', '.join(args_parts)})"

def merge_and_generate_api_strings(domain: str, base_args: Dict[str, str], pref_slot_map: Dict[str, List[str]]) -> List[str]:
    """
    Combines base arguments (from multi-turn template) with preference arguments (from history).
    """
    if not pref_slot_map:
        return [generate_single_api_string(domain, base_args)]

    # Sort keys for consistency
    sorted_pref_keys = sorted(pref_slot_map.keys())
    pref_values_lists = [pref_slot_map[k] for k in sorted_pref_keys]
    
    # Cartesian product of preference values
    pref_combinations = list(itertools.product(*pref_values_lists))
    
    results = []
    for combo in pref_combinations:
        # Start with base args
        current_args = base_args.copy()
        
        # Merge preference args
        for key, val in zip(sorted_pref_keys, combo):
            current_args[key] = val
            
        results.append(generate_single_api_string(domain, current_args))
        
    return results

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Multi-turn GT Logic)
# ---------------------------------------------------------
def format_multiturn_dialogue(query_list: List[Dict[str, str]]) -> str:
    """
    Converts list of turns into a single string.
    User: ...
    Assistant: ...
    """
    lines = []
    for turn in query_list:
        role = turn.get("role", "User")
        msg = turn.get("message", "")
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)

def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    multiturn_data: Dict[str, Any], 
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, List[str]]]: 
    
    results = []

    def to_str(val):
        if isinstance(val, bool):
            return "True" if val else "False"
        return str(val)

    # -------------------------------------------------------
    # Shared Helper: Process Extraction & Merging
    # -------------------------------------------------------
    def process_extraction(domain, extracted_pref_map):
        if domain not in multiturn_data: return None
        
        # 1. Get Multi-turn Template (Taking first example)
        template_data = multiturn_data[domain][0]
        
        # 2. Construct User Utterance (Full Dialogue)
        dialogue_text = format_multiturn_dialogue(template_data.get("query", []))
        
        # 3. Get Base API Call & Args
        base_api_str = template_data.get("api_call", [""])[0]
        _, base_args = parse_api_call_to_dict(base_api_str)
        
        # 4. Merge Base Args + extracted_pref_map
        gt_strings = merge_and_generate_api_strings(domain, base_args, extracted_pref_map)
        
        return (dialogue_text, gt_strings)

    # -------------------------------------------------------
    # [CASE 1] easy
    # -------------------------------------------------------
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                domain, current_args = parse_api_call_to_dict(call_str)
                
                if domain not in multiturn_data or domain not in pref_list: continue

                target_pref_slots = pref_list.get(domain, [])
                current_pref_map = {}
                for slot, value in current_args.items():
                    if slot in target_pref_slots:
                        current_pref_map[slot] = [to_str(value)]
                
                if current_pref_map:
                    res = process_extraction(domain, current_pref_map)
                    if res: results.append(res)

    # -------------------------------------------------------
    # [CASE 2] medium
    # -------------------------------------------------------
    elif pref_type == "medium":
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name not in pref_group_data: continue
            
            group_rules = pref_group_data[group_name].get("rules", [])
            domain_data_map = {} # { domain: { slot: [values] } }

            for evidence in pref.get("evidence", []):
                e_domain = evidence.get("domain")
                e_slot = evidence.get("slot")
                if not e_domain or not e_slot or e_domain not in multiturn_data: continue

                candidate_values = []
                for rule in group_rules:
                    if rule.get("domain") == e_domain and rule.get("slot") == e_slot:
                        candidate_values.append(to_str(rule.get("value")))
                
                if candidate_values:
                    if e_domain not in domain_data_map:
                        domain_data_map[e_domain] = {}
                    if e_slot not in domain_data_map[e_domain]:
                        domain_data_map[e_domain][e_slot] = set()
                    
                    for v in candidate_values:
                        domain_data_map[e_domain][e_slot].add(v)
            
            for domain, slot_map in domain_data_map.items():
                final_slot_map = {k: list(v) for k, v in slot_map.items()}
                res = process_extraction(domain, final_slot_map)
                if res: results.append(res)
                        
    # -------------------------------------------------------
    # [CASE 3] hard
    # -------------------------------------------------------
    elif pref_type == "hard":
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)
        
        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []

        for pref in prefs:
            current_group_name = pref.get("value_group")
            if not current_group_name or current_group_name not in pref_group_data: continue
            
            used_domains = {e.get("domain") for e in pref.get("evidence", []) if e.get("domain")}
            rules = pref_group_data[current_group_name].get("rules", [])
            
            candidate_domains = set()
            for rule in rules:
                d = rule.get("domain")
                if d and d in multiturn_data and d not in used_domains:
                    candidate_domains.add(d)
            
            for cand_domain in candidate_domains:
                slot_values_map = {}
                for rule in rules:
                    if rule.get("domain") == cand_domain:
                        s = rule.get("slot")
                        v = rule.get("value")
                        if s and v is not None:
                            if s not in slot_values_map: slot_values_map[s] = []
                            val_str = to_str(v)
                            if val_str not in slot_values_map[s]:
                                slot_values_map[s].append(val_str)
                
                if slot_values_map:
                    res = process_extraction(cand_domain, slot_values_map)
                    if res: results.append(res)

    return results

# ---------------------------------------------------------
# 3. Helpers for History & Prompt
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
# 4. vLLM API Call Wrapper
# ---------------------------------------------------------
async def call_vllm_api(
    client: AsyncOpenAI,
    prompt: str, 
    model_name: str, 
    tools_schema: List[Dict] = None,
) -> str:
    try:
        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools_schema if tools_schema else None,
            "tool_choice": "auto" if tools_schema else None,
            "temperature": 0.0,
        }

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
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
            output = message.content or ""
            
        return output.strip()
    except Exception as e:
        return f"API_ERROR: {str(e)}"

# ---------------------------------------------------------
# 5. Async Worker & Pipeline
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
    client: AsyncOpenAI,
    tools_schema: List[Dict],
    log_path: str,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    pbar: tqdm.tqdm,
):
    async with semaphore:
        current_ex = copy.deepcopy(original_ex)
        current_ex["user_utterance"] = utterance
        current_ex["reference_ground_truth"] = ground_truth
        current_ex["example_id_sub"] = f"{current_ex.get('example_id', 'unknown')}_{sub_idx}"
        
        prompt = build_input_prompt(current_ex, utterance, prompt_template, context_type, tools_schema)
        llm_output = await call_vllm_api(client, prompt, model_name, tools_schema)

        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": current_ex["example_id"],
            "example_id_sub": current_ex["example_id_sub"], 
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "injected_utterance": utterance, # Multi-turn dialogue string
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

async def process_vllm_pipeline(
    input_path: str, output_path: str, log_path: str, 
    multiturn_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, vllm_url: str, concurrency: int
):
    df = load_chains_dataset(input_path)
    
    # Load Multi-turn data instead of Query Map
    multiturn_data = load_multiturn_data(multiturn_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    
    # Initialize vLLM client (OpenAI Compatible)
    client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
    
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    
    prepared_items = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()
        if pref_type == "easy" and not original_ex.get("api_calls"): 
            skipped_count += 1; continue
        if pref_type in ["medium", "hard"] and not original_ex.get("api_calls_pref"): 
            skipped_count += 1; continue

        # Use Multi-turn Logic assignment
        pairs_list = assign_user_utterances(pref_list_path, original_ex, multiturn_data, pref_type, pref_group_path)
        
        if not pairs_list:
            skipped_count += 1; continue

        for sub_idx, (utterance, ground_truth) in enumerate(pairs_list):
            prepared_items.append({
                "original_ex": original_ex, "utterance": utterance, 
                "ground_truth": ground_truth, "sub_idx": sub_idx
            })

    print(f"Total tasks: {len(prepared_items)}. Skipped: {skipped_count}")
    pbar = tqdm.tqdm(total=len(prepared_items), desc="vLLM Inference")
    
    tasks = [process_single_item(
        item['original_ex'], item['utterance'], item['ground_truth'], item['sub_idx'],
        model_name, prompt_template, context_type, prompt_type_name, pref_type,
        client, tools_schema, log_path, semaphore, file_lock, pbar
    ) for item in prepared_items]
    
    processed_data = await asyncio.gather(*tasks)
    pbar.close()
    await client.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")

# ---------------------------------------------------------
# 6. Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output_vllm.json")
    parser.add_argument("--log_path", type=str, default="process_vllm.log")
    
    # Changed from query_path to multiturn_path
    parser.add_argument("--multiturn_path", type=str, required=True)
    
    parser.add_argument("--pref_list_path", type=str, required=True)
    parser.add_argument("--pref_group_path", type=str, required=True)
    parser.add_argument("--tools_schema_path", type=str, required=True)
    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, default="imp-zs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=50)

    from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE

    args = parser.parse_args()
    asyncio.run(process_vllm_pipeline(
        args.input_path, args.output_path, args.log_path, args.multiturn_path, 
        args.pref_list_path, args.pref_group_path, args.tools_schema_path,
        IMPLICIT_ZS_PROMPT_TEMPLATE, args.prompt_type, args.context_type, 
        args.pref_type, args.model_name, args.vllm_url, args.concurrency
    ))