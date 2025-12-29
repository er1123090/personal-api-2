import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re
import copy
import asyncio 
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

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath): return {}
    with open(fpath, "r", encoding="utf-8") as f: return json.load(f)

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance (Dataset Processing)
# ---------------------------------------------------------
def assign_user_utterances(
    pref_list_path: str, 
    example: Dict[str, Any], 
    query_map: Dict[str, str], 
    pref_type: str, 
    pref_group_path: str = None
) -> List[Tuple[str, Any]]: 
    
    results = []

    # Helper function to ensure values are strings
    def to_str(val):
        if isinstance(val, bool):
            return "True" if val else "False"
        return str(val)

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                # Parse domain and args
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in query_map or domain not in pref_list: continue

                # Regex to extract key-value pairs
                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                ground_truth_objs = []
                for slot, value in matches:
                    if slot in target_pref_slots:
                        ground_truth_objs.append({
                            "domain": domain,
                            "slot": slot,
                            "value": [to_str(value)] # Single value wrapped in list
                        })

                if ground_truth_objs:
                    # Return the list directly
                    results.append((query_map[domain], ground_truth_objs))
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
        # (Medium case logic requires excluding easy domains if necessary, 
        # but structured output logic focuses on creating the list object)
        api_calls = example.get("api_calls", [])
        easy_domain_list = []
        if isinstance(api_calls, list):
            for call_str in api_calls:
                easy_domain_list.append(call_str.split("(")[0].strip() if "(" in call_str else call_str.strip())

        prefs = example.get("api_calls_pref", [])
        if not isinstance(prefs, list) or not prefs: return []
        if not pref_group_path or not os.path.exists(pref_group_path): return []
        
        with open(pref_group_path, "r", encoding="utf-8") as f: pref_group_data = json.load(f)

        for pref in prefs:
            group_name = pref.get("value_group")
            if group_name in pref_group_data:
                seen_domains = set() 
                for evidence in pref.get("evidence", []):
                    domain = evidence.get("domain")
                    slot = evidence.get("slot")
                    values_list = evidence.get("values", [])
                    
                    if domain and (domain in query_map) and (domain not in seen_domains):
                        # Collect values
                        collected_values = []
                        for val_obj in values_list:
                            val = val_obj.get("value")
                            if val is not None:
                                collected_values.append(to_str(val))
                        
                        if collected_values:
                            ground_truth_objs = [{
                                "domain": domain,
                                "slot": slot,
                                "value": collected_values
                            }]
                            results.append((query_map[domain], ground_truth_objs))
                            seen_domains.add(domain)
        return results
        
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
            rules = pref_group_data[current_group_name].get("rules", [])
            
            # 1. Identify candidate domains
            candidate_domains = set()
            for rule in rules:
                d = rule.get("domain")
                if d and d in query_map and d not in used_domains:
                    candidate_domains.add(d)
            
            # 2. Aggregate values per slot for each candidate domain
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
                
                ground_truth_objs = []
                for s, v_list in slot_values_map.items():
                    ground_truth_objs.append({
                        "domain": cand_domain,
                        "slot": s,
                        "value": v_list 
                    })
                
                if ground_truth_objs:
                    # Return the list directly
                    results.append((query_map[cand_domain], ground_truth_objs))

        return results

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

    schema_str = ""
    if tools_schema:
        schema_str = json.dumps(tools_schema, indent=2, ensure_ascii=False)
    else:
        schema_str = "No specific schema provided."

    try:
        # 템플릿에 preference_schema가 있는 경우
        prompt = template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip(),
            preference_schema=schema_str
        )
    except KeyError:
        # 없는 경우
        prompt = template.format(
            dialogue_history=final_context,
            user_utterance=current_user_utterance.strip()
        )
    return prompt

# ---------------------------------------------------------
# 4. vLLM API Call Wrapper (Updated with DeepSeek Support)
# ---------------------------------------------------------
async def call_vllm_api(
    client: AsyncOpenAI,
    prompt: str, 
    model_name: str, 
    tools_schema: List[Dict] = None,
) -> Dict[str, str]:
    """
    Returns a dictionary:
    {
        "llm_output": str (Final Function Call),
        "reasoning_content": str (DeepSeek Thinking Process),
        "raw_content": str (Original Raw Response)
    }
    """
    llm_output = ""
    reasoning_content = ""
    raw_content = ""

    try:
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            # vLLM이 OpenAI API 형식을 따르므로 tools 사용 가능
            "tools": tools_schema if tools_schema else None,
            "tool_choice": "auto" if tools_schema else None,
            "temperature": 0.0,
        }

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        raw_content = message.content or ""
        
        # ---------------------------------------------------------
        # [HYBRID PARSING LOGIC with DeepSeek Support]
        # ---------------------------------------------------------
        
        # 1. DeepSeek <think> Token Extraction
        end_tag = "</think>"
        end_idx = raw_content.rfind(end_tag)
        
        if end_idx != -1:
            # [Reasoning] </think> 앞에 있는 모든 내용
            reasoning_part = raw_content[:end_idx]
            reasoning_content = reasoning_part.replace("<think>", "").strip()
            
            # [Clean Content] </think> 뒤에 있는 모든 내용
            clean_content = raw_content[end_idx + len(end_tag):].strip()
        else:
            reasoning_content = ""
            clean_content = raw_content.strip()

        # 2. Tool Call이 구조화되어 반환된 경우 (우선순위 1)
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
                args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                llm_output = f"{func_name}({', '.join(args_str_list)})"
            except:
                llm_output = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
        
        # 3. 텍스트 파싱 (우선순위 2 - DeepSeek R1 등 tool_call 미사용 시)
        else:
            # clean_content에서 함수 호출 패턴 추출
            match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
            if match:
                llm_output = match.group(0).strip()
            else:
                llm_output = clean_content if clean_content else "ERROR_NO_FUNC_CALL"
            
    except Exception as e:
        llm_output = f"API_ERROR: {str(e)}"

    return {
        "llm_output": llm_output.strip(),
        "reasoning_content": reasoning_content,
        "raw_content": raw_content
    }

# ---------------------------------------------------------
# 5. Async Worker
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
        current_ex["model_name"] = model_name
        
        prompt = build_input_prompt(
            current_ex, 
            current_user_utterance=utterance, 
            template=prompt_template, 
            context_type=context_type,
            tools_schema=tools_schema 
        )

        # [UPDATED] call_vllm_api returns a dict now
        result_data = await call_vllm_api(
            client=client,
            prompt=prompt, 
            model_name=model_name, 
            tools_schema=tools_schema
        )

        llm_output = result_data["llm_output"]
        reasoning_content = result_data["reasoning_content"]
        raw_content = result_data["raw_content"]

        log_record = {
            "timestamp": datetime.now().isoformat(),
            "example_id": current_ex["example_id"],
            "example_id_sub": current_ex["example_id_sub"], 
            "model_name": model_name,
            "prompt_type": prompt_type_name,
            "context_type": context_type,
            "pref_type": pref_type,
            "injected_utterance": utterance,
            "reference_ground_truth": ground_truth, 
            "model_input": prompt,
            "model_output": llm_output,
            "reasoning_tokens": reasoning_content, # [NEW]
            "raw_content": raw_content,            # [NEW]
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        current_ex["llm_output"] = llm_output
        current_ex["reasoning_tokens"] = reasoning_content # [NEW]
        current_ex["raw_content"] = raw_content            # [NEW]
        
        pbar.update(1)
        return current_ex

# ---------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------
async def process_vllm_pipeline(
    input_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, vllm_url: str, concurrency: int
):
    # Load Data
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    tools_schema = load_tools_from_file(tools_schema_path)
    
    if not tools_schema:
        print("[Warning] Tools schema is empty. 'tool_choice' will be disabled.")

    # Initialize vLLM Client
    print(f"[Info] Connecting to vLLM Server at: {vllm_url}")
    # vLLM은 dummy api key만 있으면 됨
    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")

    skipped_count = 0
    print(f"Starting ASYNC process... (Model: {model_name})")

    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    
    # Prepare Tasks
    print("Preparing tasks...")
    prepared_items = []
    
    for _, row in df.iterrows():
        original_ex = row.to_dict()

        if pref_type == "easy":
            if not original_ex.get("api_calls"): skipped_count += 1; continue
        elif pref_type in ["medium", "hard"]:
            if not original_ex.get("api_calls_pref"): skipped_count += 1; continue

        pairs_list = assign_user_utterances(pref_list_path, original_ex, query_map, pref_type, pref_group_path)
        
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
    print(f"Total tasks: {total_tasks}. Skipped source examples: {skipped_count}")

    pbar = tqdm.tqdm(total=total_tasks, desc="vLLM Inference")
    tasks = []

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
                client=client,
                tools_schema=tools_schema,
                log_path=log_path,
                semaphore=semaphore,
                file_lock=file_lock,
                pbar=pbar,
            )
        )
        tasks.append(task)
    
    processed_data = await asyncio.gather(*tasks)
    pbar.close()
    await client.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved -> {output_path}")

# ---------------------------------------------------------
# 7. Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated vLLM Inference Script")
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/data/high_regroup_dev_6.json")
    parser.add_argument("--output_path", type=str, default="output_vllm.json")
    parser.add_argument("--log_path", type=str, default="process_vllm.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/query_singleturn.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_group.json")
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/schema_easy.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group", "imp-pref"], default="imp-zs")
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name deployed on vLLM")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=50)

    # Import Prompt Template locally if available, else use placeholder
    try:
        from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE
    except ImportError:
        IMPLICIT_ZS_PROMPT_TEMPLATE = "{dialogue_history}\n{user_utterance}"

    args = parser.parse_args()

    # Select Template
    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    else: selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_vllm_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
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