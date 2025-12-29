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

# Gemini Library Import
from google import genai
from google.genai import types

# If prompt module is local, keep it; otherwise define dummy variables.
try:
    from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE
except ImportError:
    IMPLICIT_ZS_PROMPT_TEMPLATE = "{dialogue_history}\n{user_utterance}\nPredict the API call based on the preference schema: {preference_schema}"

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
# 1. Load Dataset & Queries
# ---------------------------------------------------------
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
# [Helper] API String Manipulation
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

def merge_and_generate_api_strings(domain: str, base_args: Dict[str, str], pref_slot_map: Dict[str, List[str]]) -> List[str]:
    """
    Combines base arguments (from multi-turn template) with preference arguments (from history).
    Base: {city: Vallejo}
    Pref: {price_range: [cheap, moderate]}
    Result: [GetRes(city="Vallejo", price_range="cheap"), GetRes(city="Vallejo", price_range="moderate")]
    """
    if not pref_slot_map:
        # If no preferences, just return the base call
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
        
        # Merge preference args (overwrite if collision, though usually slots differ)
        for key, val in zip(sorted_pref_keys, combo):
            current_args[key] = val
            
        results.append(generate_single_api_string(domain, current_args))
        
    return results

def generate_single_api_string(domain: str, args_dict: Dict[str, str]) -> str:
    sorted_keys = sorted(args_dict.keys())
    args_parts = [f'{key}="{args_dict[key]}"' for key in sorted_keys]
    return f"{domain}({', '.join(args_parts)})"


# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Multi-turn Logic)
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

    # Helper function to ensure values are strings
    def to_str(val):
        if isinstance(val, bool):
            return "True" if val else "False"
        return str(val)

    # -------------------------------------------------------
    # Shared Helper: Process Extraction & Merging
    # -------------------------------------------------------
    def process_extraction(domain, extracted_pref_map):
        """
        domain: e.g., "GetRestaurants"
        extracted_pref_map: e.g., {"price_range": ["cheap"]}
        """
        # 1. Get Multi-turn Template
        if domain not in multiturn_data:
            return None
        
        # We assume taking the first template example for the domain
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

                # Check if domain exists in our definitions
                if domain not in multiturn_data or domain not in pref_list: continue

                # Identify which args are actually preferences
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

            # Find matching evidence in user history
            for evidence in pref.get("evidence", []):
                e_domain = evidence.get("domain")
                e_slot = evidence.get("slot")
                if not e_domain or not e_slot or e_domain not in multiturn_data: continue

                # Collect values from the group rule that match this domain/slot
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
            
            # Generate results
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
            
            # Domains in rule BUT NOT in evidence (Implication)
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
                            if s not in slot_values_map:
                                slot_values_map[s] = []
                            val_str = to_str(v)
                            if val_str not in slot_values_map[s]:
                                slot_values_map[s].append(val_str)
                
                if slot_values_map:
                    res = process_extraction(cand_domain, slot_values_map)
                    if res: results.append(res)

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

    # NOTE: current_user_utterance is now the full Multi-turn Dialogue
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
# 5. Call LLM API
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None,
    reasoning_effort: str = None
) -> str:
    try:
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
                    return f"{func_name}({', '.join(args_str_list)})"
                except:
                    return f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
            else:
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
                return match.group(0).strip() if match else clean_content

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
        # utterance here is the full Multi-turn Dialogue string
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
            "injected_utterance": utterance, # Multi-turn dialogue
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
    multiturn_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None 
):
    df = load_chains_dataset(input_path)
    
    # Load Multi-turn data instead of Query Map
    multiturn_data = load_multiturn_data(multiturn_path)
    
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
    
    prepared_items = []
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()

        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        # Changed to pass multiturn_data
        pairs_list = assign_user_utterances(
            pref_list_path, 
            original_ex, 
            multiturn_data, 
            pref_type, 
            pref_group_path
        )
        
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
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments4/data/1229_dev_6.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    
    # Changed argument name for clarity
    parser.add_argument("--multiturn_path", type=str, default="/data/minseo/experiments4/query_multiturn-domain.json", help="Path to multiturn json file")
    
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/experiments4/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/experiments4/schema_easy.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group", "imp-pref"], default="imp-zs")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None, help="Set reasoning effort")

    args = parser.parse_args()

    selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            multiturn_path=args.multiturn_path,
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