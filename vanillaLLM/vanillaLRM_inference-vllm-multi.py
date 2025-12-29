import tqdm
import os
import json 
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import copy
import asyncio 
from openai import AsyncOpenAI 

# Gemini Library Import
import google.generativeai as genai

# If prompt module is local, keep it; otherwise define dummy variables.
try:
    from prompt import IMPLICIT_ZS_PROMPT_TEMPLATE, IMPLICIT_FS_PROMPT_TEMPLATE, IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
except ImportError:
    # Fallback templates if module is missing
    IMPLICIT_ZS_PROMPT_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"
    IMPLICIT_FS_PROMPT_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"
    IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE = "{dialogue_history}\nUser: {user_utterance}"

# Initialize API Keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)

# ---------------------------------------------------------
# [Helper] Load Tools from File (Added)
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
) -> List[Tuple[str, str]]:
    results = []

    # [CASE 1] easy
    if pref_type == "easy":
        if not os.path.exists(pref_list_path): return []
        with open(pref_list_path, "r", encoding="utf-8") as f: pref_list = json.load(f)

        api_calls = example.get("api_calls", [])
        if isinstance(api_calls, list):
            for call_str in api_calls:
                if "(" in call_str:
                    domain = call_str.split("(")[0].strip()
                    try: args_content = call_str.split("(", 1)[1].rsplit(")", 1)[0]
                    except IndexError: continue 
                else:
                    domain = call_str.strip(); args_content = ""

                if domain not in query_map or domain not in pref_list: continue

                pattern = r'(\w+)=["\']([^"\']+)["\']'
                matches = re.findall(pattern, args_content)
                target_pref_slots = pref_list.get(domain, [])
                
                if any(slot in target_pref_slots for slot, _ in matches):
                    filtered_slots = [f'{slot}="{value}"' for slot, value in matches]
                    if filtered_slots:
                        results.append((query_map[domain], f"{domain}({', '.join(filtered_slots)})"))
        return results

    # [CASE 2] medium
    elif pref_type == "medium":
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
                    if domain not in easy_domain_list and domain and (domain in query_map):
                        slots_str_list = [f'{evidence["slot"]}="{evidence["value"]}"']
                        results.append((query_map[domain], f"{domain}({', '.join(slots_str_list)})"))
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

            for rule in pref_group_data[current_group_name].get("rules", []):
                candidate_domain = rule.get("domain")
                if candidate_domain and (candidate_domain in query_map) and (candidate_domain not in used_domains):
                    target_value = rule.get("value")
                    val_str = "True" if isinstance(target_value, bool) and target_value else "False" if isinstance(target_value, bool) else str(target_value)
                    results.append((query_map[candidate_domain], f'{candidate_domain}({rule.get("slot")}="{val_str}")'))
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
def build_input_prompt(example: Dict[str, Any], current_user_utterance: str, template: str, context_type: str) -> str:
    api_str = get_api_calls_string(example)
    history_str = get_dialogue_history_string(example)
    
    final_context = ""
    if context_type == "diag-apilist":
        final_context = f"\n--- Dialogue History ---\n{history_str}\n\n--- Past API Calls ---\n{api_str}\n"
    elif context_type == "apilist-only":
        final_context = api_str
    elif context_type == "diag-only":
        final_context = history_str

    prompt = template.format(
        dialogue_history=final_context,
        user_utterance=current_user_utterance.strip(),
    )
    return prompt

# ---------------------------------------------------------
# 5. Call LLM API (Modified to use Tools)
# ---------------------------------------------------------
async def call_llm_api_async(
    prompt: str, 
    model_name: str, 
    openai_client: AsyncOpenAI = None, 
    tools_schema: List[Dict] = None
) -> Dict[str, Any]:
    """
    Returns a dictionary: 
    {
        "output": str,      # Final response text
        "reasoning": str,   # Reasoning/Thought content (if available)
        "error": str        # Error message (if any)
    }
    """
    
    baseline_prompt_path = "/data/minseo/personal-tool/conv_api/experiments3//new_baseline_prompt_update.txt"
    try:
        with open(baseline_prompt_path, "r", encoding="utf-8") as f:
            baseline_prompt = f.read()
    except FileNotFoundError:
        baseline_prompt = "You are a helpful assistant."

    result = {"output": "", "reasoning": None, "error": None}

    try:
        # --- GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key: 
                result["error"] = "API_KEY_MISSING_GOOGLE"
                return result
            
            # Gemini Model Init
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=baseline_prompt
            )
            
            # API Call
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            
            # [Reasoning Extraction Logic]
            # Gemini 2.0 Thinking models return thoughts in parts with a 'thought' attribute 
            # or as distinct text parts depending on the SDK version.
            full_text = ""
            reasoning_text = ""
            
            try:
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        # Check if the part corresponds to 'thought' (Thinking models)
                        # Note: Check SDK documentation as attribute names can vary (e.g., part.thought)
                        if hasattr(part, 'thought') and part.thought:
                             reasoning_text += str(part.thought) + "\n"
                        elif hasattr(part, 'text'):
                            full_text += part.text
                    
                    # Fallback: If no parts logic handled it, use response.text
                    if not full_text and not reasoning_text:
                        full_text = response.text
                else:
                    full_text = ""
            except Exception as parse_e:
                # Fallback mechanism if part parsing fails
                full_text = response.text if hasattr(response, 'text') else str(response)
                print(f"[Warning] Gemini parsing issue: {parse_e}")

            result["output"] = full_text.strip()
            if reasoning_text:
                result["reasoning"] = reasoning_text.strip()
            
            return result

        # --- OPENAI LOGIC ---
        else:
            if not openai_client: 
                result["error"] = "API_KEY_MISSING_OPENAI"
                return result
            
            response = await openai_client.chat.completions.create(
                model=model_name, 
                messages=[
                    # System prompt logic...
                    {"role": "user", "content": prompt},
                ],
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
            )
            
            message = response.choices[0].message
            raw_content = message.content or ""
            
            # OpenAI o1 models output reasoning usage tokens, but not the text itself usually.
            # If you are parsing <think> tags from other models (e.g. DeepSeek via API), handle here.
            # For now, we return raw_content.
            
            # [Parsing Logic]
            output_str = ""
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                    args_str_list = [f'{k}="{v}"' for k, v in func_args.items()]
                    output_str = f"{func_name}({', '.join(args_str_list)})"
                except:
                    output_str = f"ERROR_JSON_PARSE: {tool_call.function.arguments}"
            else:
                # Simple parsing to separate <think> tags if present in text (e.g. open source models)
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                
                # Check for explicit <think> block to save as reasoning
                think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
                if think_match:
                    result["reasoning"] = think_match.group(1).strip()

                match = re.search(r"([a-zA-Z0-9_]+)\((.*?)\)", clean_content, flags=re.DOTALL)
                if match:
                    output_str = match.group(0).strip()
                else:
                    output_str = clean_content
            
            result["output"] = output_str
            return result

    except Exception as e:
        print(f"LLM API Error ({model_name}): {e}")
        result["error"] = f"API_ERROR: {str(e)}"
        return result


# ---------------------------------------------------------
# 6. Pipeline (Async Version) - Modified
# ---------------------------------------------------------
async def process_single_item(
    original_ex: Dict[str, Any],
    utterance: str, 
    ground_truth: str,
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
            context_type=context_type
        )

        # [MODIFIED] Receive Dict instead of String
        llm_result = await call_llm_api_async(prompt, model_name, openai_client, tools_schema)
        
        llm_output = llm_result.get("output", "")
        llm_reasoning = llm_result.get("reasoning", None)
        error_msg = llm_result.get("error", None)

        if error_msg:
            llm_output = error_msg

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
            "model_reasoning": llm_reasoning, # [ADDED] Log reasoning separately
        }
        
        async with file_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath: os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        current_ex["llm_output"] = llm_output
        current_ex["llm_reasoning"] = llm_reasoning # [ADDED] Save to final output
        
        pbar.update(1)
        return current_ex

async def process_with_llm_async(
    input_path: str, output_path: str, log_path: str, 
    query_map_path: str, pref_list_path: str, pref_group_path: str,
    tools_schema_path: str, # [Added]
    prompt_template: str, prompt_type_name: str, context_type: str, pref_type: str,
    model_name: str, concurrency: int = 10
):
    df = load_chains_dataset(input_path)
    query_map = load_query_map(query_map_path)
    
    # [Added] Load Tools Schema
    tools_schema = load_tools_from_file(tools_schema_path)
    if not tools_schema and "gemini" not in model_name.lower():
        print("Warning: Tool schema is empty or missing.")

    openai_client = None
    if openai_api_key:
        openai_client = AsyncOpenAI(api_key=openai_api_key)

    skipped_count = 0
    print(f"Starting ASYNC process... (Model: {model_name})")

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
                tools_schema=tools_schema, # [Pass Schema]
                log_path=log_path,
                semaphore=semaphore,
                file_lock=file_lock,
                pbar=pbar
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
    
    parser.add_argument("--input_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/data/dev_4.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--log_path", type=str, default="process.log")
    parser.add_argument("--query_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/temp_queries.json")
    parser.add_argument("--pref_list_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_list.json")
    parser.add_argument("--pref_group_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/pref_group.json")
    
    # [Added] Tools Schema Argument
    parser.add_argument("--tools_schema_path", type=str, default="/data/minseo/personal-tool/conv_api/experiments3/tools_schema.json")

    parser.add_argument("--context_type", type=str, choices=["diag-apilist", "apilist-only", "diag-only"], default="diag-apilist")
    parser.add_argument("--pref_type", type=str, choices=["medium", "easy", "hard"], required=True)
    parser.add_argument("--prompt_type", type=str, choices=["imp-zs", "imp-fs", "imp-pref-group"], default="imp-zs")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--concurrency", type=int, default=20)

    args = parser.parse_args()

    if args.prompt_type == "imp-zs": selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-fs": selected_template = IMPLICIT_FS_PROMPT_TEMPLATE
    elif args.prompt_type == "imp-pref-group": selected_template = IMPLICIT_ZS_PROMPT_PREFGROUP_TEMPLATE
    else: selected_template = IMPLICIT_ZS_PROMPT_TEMPLATE

    asyncio.run(
        process_with_llm_async(
            input_path=args.input_path,
            output_path=args.output_path,
            log_path=args.log_path,
            query_map_path=args.query_path,
            pref_list_path=args.pref_list_path,
            pref_group_path=args.pref_group_path,
            tools_schema_path=args.tools_schema_path, # [Pass Path]
            prompt_template=selected_template,
            prompt_type_name=args.prompt_type,
            context_type=args.context_type,
            pref_type=args.pref_type,
            model_name=args.model_name,
            concurrency=args.concurrency
        )
    )