import json
import os
import time
import copy
import httpx
from openai import OpenAI

# ================= 1. Configuration Area =================

# Lab OpenAI Proxy
BASE_URL = "http://35.220.164.252:3888/v1"
http_client = httpx.Client(timeout=60.0, verify=False)
API_KEY = "sk-aEgLZS952k0LiNx7kJYJOo5tkhOERlNQfxhnV5xIugxAzltm"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# Directories
PROMPT_ROOT_DIR = '/mnt/shared-storage-user/leijiayi/counterfactual/prompt/physics'
# 全新的输出目录，直接生成最终版数据，无需依赖 v4
NEW_EVAL_ROOT_DIR = '/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5_direct/physics' 

# Target Subjects Filter
RAW_TARGET_SUBJECTS = os.environ.get("TARGET_SUBJECTS", "").strip()
TARGET_SUBJECTS = {s.strip() for s in RAW_TARGET_SUBJECTS.split(",") if s.strip()} if RAW_TARGET_SUBJECTS else None

MODEL_NAME = "gemini-3-pro-preview-thinking"

# ================= 2. Unified Prompt Templates =================

# 全新的综合 Prompt，融合了旧版的 D1/D3 和新版的严苛 D2
PROMPT_TEMPLATE = """# Role: AI Image Quality Assurance Specialist

## Task
You are an expert in evaluating Text-to-Image (T2I) generation models.
I will provide you with an Input Prompt and its Assessment Points.
Your goal is to generate a Strict Evaluation Protocol consisting of specific questions covering 3 dimensions.

## Input Data
**Input Prompt**: {input_prompt}
**Assessment Points**: {assessment_points}

## Guidelines for Question Generation

### Dimension 1: Visual Integrity
*   **Focus**: Technical image quality (sharpness, anatomy). Style-agnostic.
*   **Weight**: Assign a weight of 2 or 3.

### Dimension 2: Assessment Point (CRITICAL)
*   **Focus**: Verify the human-written `Assessment Points`.
*   **CRITICAL RULE 1**: Combine all assessment points into a **SINGLE, comprehensive question**. Do NOT split into multiple questions.
*   **CRITICAL RULE 2**: Make the 0.5 score extremely hard to get. 0.0 means the criteria are not met.
*   **Weight**: Assign a weight of 15.

{dimension_3_section}

## Output Format
Return **ONLY** a valid JSON List. No markdown formatting (```json).
"""

D3_FACTUAL = """### Dimension 3: Counterfactual Logic (Factual L1)
*   **Focus**: Verify that the image adheres strictly to standard real-world physics and logic.
*   **Strict Scoring Rule**: 1.0 = Flawless physics. 0.5 = Minor logical flaw. 0.0 = Clear violation of physics (e.g., floating objects). Do NOT give partial credit just because the main subject is present.
*   **Weight**: Assign a weight of 8."""

D3_COUNTERFACTUAL = """### Dimension 3: Global Counterfactual Consistency (L2/L3)
*   **Focus**: Verify if the **ENTIRE SCENE** strictly adheres to the counterfactual premise demanded by the prompt.
*   **Strict Scoring Rule**: Zero Tolerance for Logical Fractures. 1.0 = Entire world follows the new rule. 0.0 = Logical Fracture (e.g., main subject is counterfactual, but environment reverts to normal physics).
*   **Weight**: Assign a weight of 8."""

# ================= 3. Core Functions =================

def call_gemini(input_prompt, assessment_points, prompt_level, max_retries=10): 
    """一次性调用大模型生成 D1, D2(合并版), D3"""
    d3_section = D3_FACTUAL if prompt_level == 'l1' else D3_COUNTERFACTUAL
    final_prompt = PROMPT_TEMPLATE.format(
        input_prompt=input_prompt,
        assessment_points=assessment_points,
        dimension_3_section=d3_section
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always return a valid JSON array, no markdown formatting."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'): content = content[7:]
            elif content.startswith('```'): content = content[3:]
            if content.endswith('```'): content = content[:-3]
            
            result = json.loads(content.strip())
            return result if isinstance(result, list) else result.get(list(result.keys())[0])
                    
        except Exception as e:
            print(f"⚠️ API Error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(60 if "429" in str(e) else 2 ** attempt)
    return None

def process_single_file(prompt_file, new_eval_file):
    print(f"\n📂 Processing: {os.path.basename(prompt_file)}")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
    except Exception:
        return

    # 断点续传逻辑
    if os.path.exists(new_eval_file):
        with open(new_eval_file, 'r', encoding='utf-8') as f:
            new_eval_data = json.load(f)
    else:
        new_eval_data = []
        
    processed_keys = {(str(item.get('source_id')), item.get('prompt_level')) for item in new_eval_data}
    next_id_counter = max([int(item.get('id', 0)) for item in new_eval_data] + [0]) + 1
    modified = False

    for item in prompt_data:
        source_id = str(item.get('id'))
        category = item.get('category', '')
        
        # 依次处理 L1 和 L2
        for level in ['l1', 'l2']:
            if (source_id, level) in processed_keys:
                continue
                
            curr_prompt = item.get(f'prompt_{level}', '')
            curr_ap = item.get(f'ap_{level}', '')
            
            if not curr_prompt or not curr_ap:
                continue
                
            print(f"   ⚡ Generating ID:{source_id} Level:{level} ...")
            questions = call_gemini(curr_prompt, curr_ap, level)
            
            if questions:
                level_records = []
                for q in questions:
                    q.update({
                        "id": str(next_id_counter),
                        "source_id": source_id,
                        "prompt_level": level,
                        "category": category,
                        "generator_model": MODEL_NAME
                    })
                    next_id_counter += 1
                    level_records.append(q)
                
                new_eval_data.extend(level_records)
                processed_keys.add((source_id, level))
                modified = True
                
                # 如果是 L2，直接复制一份作为 L3
                if level == 'l2':
                    l3_records = copy.deepcopy(level_records)
                    for q in l3_records:
                        q['id'] = str(next_id_counter)
                        q['prompt_level'] = 'l3'
                        next_id_counter += 1
                    new_eval_data.extend(l3_records)
                    processed_keys.add((source_id, 'l3'))
                
                # 实时保存，防止中断
                os.makedirs(os.path.dirname(new_eval_file), exist_ok=True)
                with open(new_eval_file, 'w', encoding='utf-8') as f:
                    json.dump(new_eval_data, f, ensure_ascii=False, indent=2)
                
                time.sleep(1) # API 缓冲

    print("   🎉 File processing complete." if modified else "   ✅ No new data to process.")

def main():
    print("🚀 Starting All-in-One Generation Pipeline...")
    for root, dirs, files in os.walk(PROMPT_ROOT_DIR):
        for file in files:
            if not file.endswith(".json"): continue
            file_name_no_ext = os.path.splitext(file)[0]
            if TARGET_SUBJECTS and file_name_no_ext not in TARGET_SUBJECTS: continue

            rel_path = os.path.relpath(os.path.join(root, file), PROMPT_ROOT_DIR)
            prompt_file = os.path.join(PROMPT_ROOT_DIR, rel_path)
            new_eval_file = os.path.join(NEW_EVAL_ROOT_DIR, os.path.dirname(rel_path), f"{file_name_no_ext}_gemini.json")
            
            process_single_file(prompt_file, new_eval_file)
    print("\n🏁 All tasks finished!")

if __name__ == "__main__":
    main()
