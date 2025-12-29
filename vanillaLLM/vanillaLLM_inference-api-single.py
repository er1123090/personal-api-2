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

def load_query_map(fpath: str) -> Dict[str, str]:
    if not os.path.exists(fpath):
        return {}
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------
# 2. Logic to Assign User Utterance
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
                ground_truth_pref_slots = pref_list.get(domain, [])
                
                ground_truth_objs = []
                for slot, value in matches:
                    if slot in ground_truth_pref_slots:
                        ground_truth_objs.append({
                            "domain": domain,
                            "slot": slot,
                            "value": [to_str(value)] 
                        })

                if ground_truth_objs:
                    # [수정됨] 딕셔너리로 감싸지 않고 리스트 자체를 append 합니다.
                    results.append((query_map[domain], ground_truth_objs))
                    
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
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
                        collected_values = []
                        for val_obj in values_list:
                            val = val_obj.get("value")
                            if val is not None:
                                collected_values.append(to_str(val))
                        
                        if collected_values:
                            # [수정됨] 리스트 형태로 바로 생성
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
            
            candidate_domains = set()
            for rule in rules:
                d = rule.get("domain")
                if d and d in query_map and d not in used_domains:
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
                
                ground_truth_objs = []
                for s, v_list in slot_values_map.items():
                    ground_truth_objs.append({
                        "domain": cand_domain,
                        "slot": s,
                        "value": v_list 
                    })
                
                if ground_truth_objs:
                    # [수정됨] 딕셔너리로 감싸지 않고 리스트 자체를 append 합니다.
                    results.append((query_map[cand_domain], ground_truth_objs))

        return results

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
    reasoning_effort: str = None  # <--- [NEW] Reasoning/Thinking parameter ("low", "high", etc.)
) -> str:
    
    # baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments4//new_baseline_prompt_update.txt"
    # baseline_prompt = "You are a helpful assistant."
    # ... (파일 읽기 로직 생략) ...

    try:
        # --- GEMINI LOGIC (NEW SDK: google-genai) ---
        if "gemini" in model_name.lower():
            if not google_api_key: return "API_KEY_MISSING_GOOGLE"
            
            # 1. Client 초기화
            client = genai.Client(api_key=google_api_key)
            
            # 2. Config 설정 준비
            config_params = {
                "temperature": 0.0,
                # "system_instruction": baseline_prompt  # 시스템 프롬프트가 필요하면 여기에 추가
            }

            # [NEW] Thinking Config 설정
            # Gemini 2.0 Flash Thinking 등 지원 모델인 경우
            if reasoning_effort and ("gemini-3" in model_name.lower() or "flash" in model_name.lower()):
                # types.ThinkingConfig 객체 생성
                # include_thoughts=True는 사고 과정을 포함할지 여부 (보통 True로 설정)
                config_params["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=reasoning_effort.lower() # "low", "high"
                )

            # 3. GenerateContentConfig 객체 생성
            conf = types.GenerateContentConfig(**config_params)
            
            # 4. API 호출 (Async)
            # 주의: 새로운 SDK에서 비동기 호출은 client.aio를 사용합니다.
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=conf
            )
            
            return response.text.strip()

        # --- OPENAI LOGIC ---
        else:
            if not openai_client: return "API_KEY_MISSING_OPENAI"
            
            # [NEW] Prepare arguments for OpenAI
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "tools": tools_schema if tools_schema else None,
                "tool_choice": "auto" if tools_schema else None,
            }

            # [NEW] Inject reasoning_effort only for supported models
            is_reasoning_model = any(k in model_name.lower() for k in ["o1", "o3", "gpt-5", "gpt-5.1", "gpt-5-mini"])
            if is_reasoning_model and reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort.lower() # "low", "medium", "high"

            response = await openai_client.chat.completions.create(**kwargs)
            
            message = response.choices[0].message
            raw_content = message.content or ""

            # [Parsing Logic]
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
                # <think> 태그 제거 로직 (DeepSeek이나 호환 모델 대비용)
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
    reasoning_effort: str = None # <--- [NEW]
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

        # [NEW] Pass reasoning_effort
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
            "injected_utterance": utterance,
            "reasoning_effort": reasoning_effort, # [NEW] Log it
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
    tools_schema_path: str,
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10,
    reasoning_effort: str = None # <--- [NEW]
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
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
                reasoning_effort=reasoning_effort # <--- [NEW]
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
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/data/high_regroup_dev_6.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/query_singleturn.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/pref_group.json")
    
    # Tools Schema Argument
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments4/schema_easy.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    # [NEW] Added 'imp-pref'
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group", "imp-pref"], default="imp-zs")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)
    
    # [NEW] Reasoning Effort Argument
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", 'low', "medium", "high"], default=None, help="Set reasoning effort for supported models (o1, gpt-5, gemini-3)")

    args = parser.parse_args()

    # Template selection
    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_with_llm_async(
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
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort # <--- [NEW]
        )
    )