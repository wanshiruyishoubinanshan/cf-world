import os
import json
import re
import traceback
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= ⚙️ Configuration Area =================

MODEL_PATH = "/path/to/models/Qwen3-VL-235B-Instruct-FP8"
EVAL_JSON_PATH = "./data/sampled_Attribute_decoupling.json" # Replace with your new JSON path
IMAGE_BASE_DIR = "./data/images"  
OUTPUT_BASE_DIR = "./output/scores_attribute" # New output directory

TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 8192
BATCH_SIZE_PER_SAVE = 32

# ================= 🛠️ Utility Functions =================

def extract_json_robust(text):
    if not text: return None
    text = re.sub(r'^```(json)?', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    try:
        p1, p2 = text.find('{'), text.rfind('}')
        if p1 != -1 and p2 != -1: return json.loads(text[p1:p2+1])
    except: pass
    return None

def calculate_scores(results_list):
    """Calculate average scores for overall, factual, and unusual"""
    scores = {'factual': [], 'unusual': []}
    
    for item in results_list:
        cat = str(item.get('category', '')).lower()
        score = float(item.get('score', 0.0))
        if cat in scores:
            scores[cat].append(score)
            
    factual_score = sum(scores['factual']) / len(scores['factual']) if scores['factual'] else 0.0
    unusual_score = sum(scores['unusual']) / len(scores['unusual']) if scores['unusual'] else 0.0
    
    all_scores = scores['factual'] + scores['unusual']
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return results_list, round(overall_score, 4), round(factual_score, 4), round(unusual_score, 4)

# ================= 🚀 Main Logic =================

def main():
    print(f"🔍 Reading evaluation questions file: {EVAL_JSON_PATH}")
    with open(EVAL_JSON_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    print(f"✅ Loaded {len(eval_data)} evaluation tasks.")

    model_names = [d for d in os.listdir(IMAGE_BASE_DIR) if os.path.isdir(os.path.join(IMAGE_BASE_DIR, d))]
    model_names.sort()
    print(f"📦 Found {len(model_names)} models to evaluate: {', '.join(model_names)}")
    
    print(f"\n⏳ Loading Qwen3-VL-235B (TP={TENSOR_PARALLEL_SIZE})...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            dtype="bfloat16",
            enforce_eager=False,
            limit_mm_per_prompt={"image": 1}, 
            distributed_executor_backend="ray",
            swap_space=0, 
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=2048, stop_token_ids=[151645, 151643])
    global_comparison_stats = []

    # Unified System Prompt (Emphasizing the presence and relationship of Entity A and B)
    unified_system_prompt = """You are an objective and strict Image Quality Assurance Judge. Your task is to evaluate whether an AI-generated image accurately reflects the provided prompt.
    
    CRITICAL EVALUATION CRITERIA:
    1. Entity A MUST be clearly visible and identifiable.
    2. Entity B MUST be clearly visible and identifiable.
    3. The relationship or interaction between Entity A and Entity B MUST exactly match the prompt.
    
    SCORING:
    - Score 1.0: Both entities are present, distinct, and their relationship perfectly matches the prompt.
    - Score 0.5: Both entities are present, but their relationship is slightly off, or one entity is partially blended/malformed.
    - Score 0.0: One or both entities are missing, severely blended together, or the relationship is completely wrong.

    You MUST output ONLY a valid JSON object. 
    CRITICAL: Generate the "reasoning" BEFORE the "score".
    Example format:
    {
      "reasoning": "Entity A (Giant Otter) is clearly visible. Entity B (St. Peter's Basilica) is in the background. The otter is swimming in a reflection pool as described. No concept blending observed.",
      "score": 1.0
    }"""

    for model_idx, current_model in enumerate(model_names):
        print(f"\n🚀 Starting to process model [{model_idx+1}/{len(model_names)}]: {current_model}")
        
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, current_model)
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, "eval_results.json")

        final_results_list = []
        processed_ids = set()

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    final_results_list = json.load(f)
                for item in final_results_list:
                    processed_ids.add(str(item.get('id')))
                print(f"⏩ Loaded existing progress: {len(processed_ids)} images")
            except Exception:
                pass

        tasks_to_run = []
        for q in eval_data:
            q_id = str(q['id'])
            if q_id in processed_ids: 
                continue
            
            # Match your directory structure: model_name/sampled_Attribute_decoupling/id.png
            img_name = f"{q_id}.png"
            img_path = os.path.join(IMAGE_BASE_DIR, current_model, "sampled_Attribute_decoupling", img_name)
            
            if not os.path.exists(img_path):
                img_path_jpg = img_path.replace(".png", ".jpg")
                if os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                else:
                    res = q.copy()
                    res.update({"score": 0.0, "reasoning": "Image missing", "evaluator": "None"})
                    final_results_list.append(res)
                    continue
            
            try:
                image_obj = Image.open(img_path).convert("RGB")
                tasks_to_run.append({
                    "q_data": q,
                    "image_obj": image_obj,
                    "retry_count": 0
                })
            except Exception as e:
                print(f"   ❌ Image reading error {img_path}: {e}")

        chunk_size = BATCH_SIZE_PER_SAVE
        total_chunks = (len(tasks_to_run) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min((chunk_idx + 1) * chunk_size, len(tasks_to_run))
            pending_in_batch = tasks_to_run[start_i:end_i]
            
            while pending_in_batch:
                batch_prompts = []
                valid_tasks = []
                
                for task in pending_in_batch:
                    q = task['q_data']
                    user_content = f"""Please evaluate the image based on the following details:
                    Prompt: "{q['prompt']}"
                    Entity A to check: "{q['A']}"
                    Entity B to check: "{q['B']}"
                    """

                    if task['retry_count'] > 0: 
                        user_content += "\n\n[ERROR] Previous response invalid. OUTPUT JSON ONLY."
                    
                    messages = [
                        {"role": "system", "content": unified_system_prompt}, 
                        {"role": "user", "content": [{"type": "image", "image": task['image_obj']}, {"type": "text", "text": user_content}]}
                    ]
                    
                    try:
                        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        batch_prompts.append({"prompt": prompt_text, "multi_modal_data": {"image": task['image_obj']}})
                        valid_tasks.append(task)
                    except: pass

                if not batch_prompts: break
                outputs = llm.generate(batch_prompts, sampling_params)
                next_round_pending = []
                
                for i, output in enumerate(outputs):
                    task = valid_tasks[i]
                    q = task['q_data']
                    parsed = extract_json_robust(output.outputs[0].text)
                    
                    if parsed and isinstance(parsed, dict):
                        final_score = float(parsed.get("score", 0.0))
                        final_reasoning = parsed.get("reasoning", "Parsed")
                        
                        res = q.copy()
                        res.update({
                            "score": final_score,
                            "reasoning": final_reasoning,
                            "evaluator": "Qwen3-VL-235B"
                        })
                        final_results_list.append(res)
                    else:
                        if task['retry_count'] < 3:
                            task['retry_count'] += 1
                            next_round_pending.append(task)
                        else:
                            res = q.copy()
                            res.update({"score": 0.0, "reasoning": "FAILED PARSING"})
                            final_results_list.append(res)
                            
                pending_in_batch = next_round_pending
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)

        if final_results_list:
            final_results_list, overall, factual, unusual = calculate_scores(final_results_list)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)
                
            global_comparison_stats.append({
                "model_name": current_model,
                "overall_score": overall,
                "factual_score": factual,
                "unusual_score": unusual
            })
            print(f"📈 {current_model} -> Overall: {overall:.4f} | Factual: {factual:.4f} | Unusual: {unusual:.4f}")

    print("\n🏆 All models processed! Generating global comparison summary table...")
    global_summary_path = os.path.join(OUTPUT_BASE_DIR, "global_models_comparison.json")
    global_comparison_stats.sort(key=lambda x: x.get('overall_score', 0.0), reverse=True)
    
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        json.dump(global_comparison_stats, f, indent=2, ensure_ascii=False)
        
    for rank, stat in enumerate(global_comparison_stats):
        print(f"  {rank+1}. {stat['model_name']:<15} | Overall: {stat['overall_score']:.4f} | Factual: {stat['factual_score']:.4f} | Unusual: {stat['unusual_score']:.4f}")

if __name__ == "__main__":
    main()
