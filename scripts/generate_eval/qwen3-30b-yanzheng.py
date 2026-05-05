import os
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer  # ✅ 确保只使用 Tokenizer

# =============== 1. 配置路径 ===============
MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct-FP8"
INPUT_PATH = "prompt_yanzheng/rule_decouple.json"
OUTPUT_FILE = "eval_yanzheng/rule_decouple_eval_questions.json"

# =============== 2. 核心 Prompt 模板 ===============
PROMPT_TEMPLATE = """# Role: AI Image Quality Assurance Specialist

## Task
You are an expert in evaluating Text-to-Image (T2I) generation models.
I will provide you with two inputs:
1. **Input Prompt**: The actual text sent to the T2I model.
2. **Assessment Points**: A list of specific criteria or constraints that must be verified.

Your goal is to generate a **Strict Evaluation Protocol** consisting of **4 to 8 specific questions**.
You must output the result as a structured JSON List.

## Input Data
**Input Prompt**:
{input_prompt}

**Assessment Points**:
{assessment_points}

## Guidelines for Question Generation & Weighting
You must generate questions covering the following 2 dimensions, strictly adhering to the weighting rules.

### Dimension 1: Visual Integrity
*   **Focus**: Technical image quality (sharpness, anatomy). Style-agnostic (cartoons are fine).
*   **Weight**: Assign a weight of **2 or 3**.

### Dimension 2: Assessment Point
*   **Focus**: Verify the human-written `Assessment Points`. 
*   **Strategy**: Split the APs into distinct questions. 
*   **CRITICAL RULE FOR CRITERIA**: Make the 0.5 score extremely hard to get. Do NOT give partial credit just because the main subject exists. 0.5 should only be given if the counterfactual action is attempted but slightly flawed. If the image reverts to normal reality, the score MUST be 0.0.
*   **Weight**: The SUM of weights here must be between **14 and 16**.

## Output Format
Return **ONLY** a valid JSON List. No markdown formatting (```json).

**JSON Structure Example:**
[
    {{
        "question_type": "Assessment Point",
        "question": "Is the cat explicitly floating in mid-air with NO support?",
        "evaluation_criteria": "1.0 – Cat is clearly hovering with no visible support or motion blur. 0.5 – Cat is in the air but looks like it is jumping/falling (motion blur present). 0.0 – Cat is touching the ground or any surface. (Just being a cat is NOT enough for partial credit).",
        "weight": 8
    }},
    {{
        "question_type": "Counterfactual Logic",
        "question": "Does the fire explicitly demonstrate cold properties (e.g., freezing things, covered in frost) rather than burning?",
        "evaluation_criteria": "1.0 – Fire is freezing objects or emitting snow/frost. 0.5 – Fire is blue/white but lacks explicit freezing effects. 0.0 – Fire is burning things, emitting smoke, or acting like normal hot fire.",
        "weight": 10
    }}
]
"""

# =============== 3. 堆栈式 JSON 提取器 ===============
def extract_json_with_stack(text):
    if not text: return None, "Empty Input"

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")

    stack = []
    start_index = -1

    for i, char in enumerate(text):
        if char == '[':
            if not stack:
                start_index = i 
            stack.append(char)
        elif char == ']':
            if stack:
                stack.pop()
                if not stack:
                    json_candidate = text[start_index : i+1]
                    try:
                        parsed = json.loads(json_candidate)
                        return parsed, "Success (Stack)"
                    except json.JSONDecodeError:
                        try:
                            fixed = re.sub(r',\s*]', ']', json_candidate)
                            parsed = json.loads(fixed)
                            return parsed, "Success (Stack+Fix)"
                        except:
                            pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), "Success (Regex)"
        except:
            pass
            
    return None, text[-300:] 

# =============== 4. 准备数据 ===============
def prepare_prompts(data, tokenizer):  # ✅ 参数改为 tokenizer
    prompts = []
    metadata = []

    print(f"正在构建输入数据...")
    for item in data:
        source_id = item.get("id")
        rule_id = item.get("rule_id")
        category = item.get("category", "cf").lower()
        input_prompt = item.get("prompt", "")
        assessment_points = item.get("ap", "None provided.")
        
        content = PROMPT_TEMPLATE.format(
            input_prompt=input_prompt, 
            assessment_points=assessment_points,
        )
        
        messages = [
            {
                "role": "system", 
                "content": "You are a strict JSON-only machine. Do NOT output any reasoning, thinking process, or conversational text. Output ONLY the JSON list starting with '[' and ending with ']'."
            },
            {"role": "user", "content": content}
        ]

        
        # ✅ 使用 tokenizer
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(text)
        metadata.append({
            "source_id": source_id, 
            "rule_id": rule_id,
            "category": category
        })
                
    return metadata, prompts

# =============== 主流程 ===============
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到输入文件: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("正在加载 Tokenizer...")
    # ✅ 彻底替换掉 AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    meta_list, prompts = prepare_prompts(data, tokenizer)

    print(f"正在加载 vLLM 模型: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        # 🚨 核心修改：397B 模型必须多卡并行！请根据您的实际 GPU 数量设为 4 或 8
        tensor_parallel_size=4, 
        gpu_memory_utilization=0.95, 
        max_model_len=8192, 
        dtype="bfloat16",
        enforce_eager=False
    )

    sampling_params = SamplingParams(
        temperature=0.3, 
        top_p=0.9,
        max_tokens=2048, 
        stop_token_ids=[151645, 151643]
    )

    print(f"🚀 开始批量推理 (共 {len(prompts)} 条)...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    success_count = 0
    global_id_counter = 1

    print("\n" + "="*50)
    print("🔍 结果检查")
    print("="*50)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = meta_list[i]
        
        parsed, status = extract_json_with_stack(generated_text)
        
        if parsed and isinstance(parsed, list) and len(parsed) > 0:
            success_count += 1
            print(f"Source ID {meta['source_id']} (Rule {meta['rule_id']}): ✅ Success ({len(parsed)} questions generated)")
            
            for q in parsed:
                results.append({
                    "id": global_id_counter, 
                    "source_id": meta['source_id'],
                    "rule_id": meta['rule_id'],
                    "category": meta['category'],
                    "question_type": q.get("question_type", "Unknown"),
                    "question": q.get("question", ""),
                    "evaluation_criteria": q.get("evaluation_criteria", ""),
                    "weight": q.get("weight", 0)
                })
                global_id_counter += 1
        else:
            print(f"Source ID {meta['source_id']} (Rule {meta['rule_id']}): ❌ Failed.")
            print(f"   Reason: {status}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 统计: 成功 {success_count}/{len(prompts)}")
    print(f"结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
