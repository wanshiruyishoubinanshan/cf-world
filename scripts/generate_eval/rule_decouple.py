import os
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============== 1. Configure Paths ===============
MODEL_PATH = ""
INPUT_PATH = "input_data/rule_decouple.json"
OUTPUT_FILE = "output_data/rule_decouple_eval_questions.json"

# =============== 2. Core Prompt Template ===============
PROMPT_TEMPLATE = """# Role: AI Image Quality Assurance Specialist

## Task
You are an expert in evaluating Text-to-Image (T2I) generation models.
I will provide you with two inputs:
1. **Input Prompt**: The actual text sent to the T2I model.
2. **Assessment Points**: A list of specific criteria or constraints that must be verified.

Your goal is to generate a **Strict Evaluation Protocol** consisting of 1 specific questions.
You must output the result as a structured JSON List.

## Input Data
**Input Prompt**:
{input_prompt}

**Assessment Points**:
{assessment_points}

*   **Focus**: Verify the human-written `Assessment Points`. 
*   **Strategy**: Split the APs into distinct questions. 
*   **CRITICAL RULE FOR CRITERIA**: Make the 0.5 score extremely hard to get. Do NOT give partial credit just because the main subject exists. 0.5 should only be given if the counterfactual action is attempted but slightly flawed. If the image reverts to normal reality, the score MUST be 0.0.
*   **Weight**: The weight here must be 15.

## Output Format
Return **ONLY** a valid JSON List. No markdown formatting (```json).

**JSON Structure Example:**
[
    {{
        "question_type": "Assessment Point",
        "question": "Does the arrow point to left?",
        "evaluation_criteria": "1.0 – The arrow is clearly and unambiguously pointing directly to the left. 0.5 – The arrow points diagonally towards the left (e.g., up-left or down-left), or the arrowhead is slightly malformed but the intended leftward direction is visible. 0.0 – The arrow points right, straight up, straight down, or no arrow is present. (Simply generating an arrow without the correct direction gets 0.0).",
        "weight": 15
    }},
]
"""

# =============== 3. Stack-based JSON Extractor ===============
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

# =============== 4. Prepare Data ===============
def prepare_prompts(data, tokenizer):  # ✅ Parameter changed to tokenizer
    prompts = []
    metadata = []

    print(f"Building input data...")
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

        # ✅ Use tokenizer
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

# =============== Main Process ===============
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading Tokenizer...")
    # ✅ Completely replaced AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    meta_list, prompts = prepare_prompts(data, tokenizer)

    print(f"Loading vLLM model: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        # 🚨 Core modification: Large models require multi-GPU parallelism! Set to 4 or 8 based on your actual GPU count
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

    print(f"🚀 Starting batch inference (Total {len(prompts)} items)...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    success_count = 0
    global_id_counter = 1

    print("\n" + "="*50)
    print("🔍 Result Check")
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

    print(f"\n🎉 Statistics: Success {success_count}/{len(prompts)}")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
