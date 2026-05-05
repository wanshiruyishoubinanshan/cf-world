import os
import json
import re
import glob
import traceback
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= ⚙️ Configuration Area =================

# 1. Model Path
MODEL_PATH = "/path/to/models/Qwen3-VL-235B-Instruct-FP8"

# 2. Path Configuration
EVAL_ROOT_DIR = "./data/eval_question"
IMAGE_BASE_DIR = "./data/output/images"
OUTPUT_BASE_DIR = "./output/scores-de-norm"

# 3. Hardware & Inference Configuration
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 8192
BATCH_SIZE_PER_SAVE = 32

# ================= 🛠️ Utility Functions =================

def extract_json_robust(text):
    """Extract JSON from LLM response"""
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

def calculate_l2_stats(results_list, filename):
    """Calculate statistical scores for L2 only"""
    scores = []
    unique_images = {}
    
    for item in results_list:
        sid = str(item.get('source_id'))
        score = float(item.get('score', 0.0))
        weight = float(item.get('weight', 1.0))
        
        if sid not in unique_images:
            unique_images[sid] = {'total_score': 0, 'total_weight': 0}
            
        unique_images[sid]['total_score'] += score * weight
        unique_images[sid]['total_weight'] += weight
        
    for sid, data in unique_images.items():
        if data['total_weight'] > 0:
            img_score = data['total_score'] / data['total_weight']
            scores.append(img_score)
            
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {"filename": filename, "l2_overall": round(avg_score, 4)}, scores

# ================= 🚀 Main Logic =================

def main():
    print(f"🔍 Scanning evaluation files: {EVAL_ROOT_DIR}/**/*.json")
    json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
    if not json_files:
        print("❌ No JSON files found.")
        return
        
    if not os.path.exists(IMAGE_BASE_DIR):
        print(f"❌ Image base directory not found: {IMAGE_BASE_DIR}")
        return
        
    # Get all model names (e.g., bagel, etc.)
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
        traceback.print_exc()
        return

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=2048, stop_token_ids=[151645, 151643])
    global_comparison_stats = []

    # ================= 🔄 Iterate through all models =================
    for model_idx, current_model in enumerate(model_names):
        print("\n" + "★"*80)
        print(f"🚀 Starting to process model [{model_idx+1}/{len(model_names)}]: {current_model}")
        print("★"*80)
        
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, current_model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Specific de-norm image directory for the current model
        model_image_dir = os.path.join(IMAGE_BASE_DIR, current_model, "de-norm")
        
        if not os.path.exists(model_image_dir):
            print(f"⚠️  Image directory not found: {model_image_dir} (Skipping this model)")
            continue

        all_model_scores = [] 
        all_files_statistics_model = []

        for json_idx, json_file in enumerate(json_files):
            rel_path = os.path.relpath(json_file, EVAL_ROOT_DIR)
            output_file = os.path.join(model_output_dir, rel_path.replace(".json", "_qwen235b.json"))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # --- 🔥 NEW: Dynamically parse subfolder path ---
            path_obj = Path(rel_path)
            parent_dir = str(path_obj.parent)
            stem = path_obj.stem.replace("_gemini", "").replace("_eval", "")
            
            if parent_dir == '.':
                specific_image_dir = os.path.join(IMAGE_BASE_DIR, current_model, "de-norm", stem)
            else:
                specific_image_dir = os.path.join(IMAGE_BASE_DIR, current_model, "de-norm", parent_dir, stem)
            # ----------------------------------

            final_results_list = []
            processed_keys = set()

            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        final_results_list = json.load(f)
                    for item in final_results_list:
                        processed_keys.add(str(item.get('source_id')))
                    print(f"⏩ Loaded existing progress: {len(processed_keys)} images")
                except Exception:
                    final_results_list = []

            with open(json_file, 'r') as f:
                eval_data = json.load(f)

            # Extract only L2 tasks
            l2_tasks_map = {}
            for item in eval_data:
                if str(item.get('prompt_level', '')).lower() == 'l2':
                    sid = str(item['source_id'])
                    if sid not in l2_tasks_map: 
                        l2_tasks_map[sid] = []
                    l2_tasks_map[sid].append(item)

            if not l2_tasks_map:
                continue

            tasks_to_run = []
            for sid, questions in l2_tasks_map.items():
                if sid in processed_keys: continue
                
                # Match {source_id}.png or .jpg directly in the corresponding subfolder
                img_path = os.path.join(specific_image_dir, f"{sid}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(specific_image_dir, f"{sid}.jpg")

                if not os.path.exists(img_path):
                    for q in questions:
                        res = q.copy()
                        res.update({"score": 0.0, "reasoning": "Image missing in de-norm", "evaluator": "None"})
                        final_results_list.append(res)
                    continue
                
                try:
                    image_obj = Image.open(img_path).convert("RGB")
                    q_text = ""
                    temp_id_map = {}
                    for idx, q in enumerate(questions):
                        temp_id = str(idx + 1)
                        temp_id_map[temp_id] = q
                        q_text += f'\nQuestion ID: "{temp_id}"\nQuestion: {q["question"]}\nCriteria: {q["evaluation_criteria"]}\n---'

                    tasks_to_run.append({
                        "source_id": sid,
                        "filename": os.path.basename(img_path),
                        "image_obj": image_obj,
                        "temp_id_map": temp_id_map,
                        "base_q_text": q_text,
                        "retry_count": 0
                    })
                except Exception as e:
                    print(f"   ❌ Image reading error {sid}: {e}")

            # Inference logic
            chunk_size = BATCH_SIZE_PER_SAVE
            total_chunks = (len(tasks_to_run) + chunk_size - 1) // chunk_size

            for chunk_idx in range(total_chunks):
                start_i = chunk_idx * chunk_size
                end_i = min((chunk_idx + 1) * chunk_size, len(tasks_to_run))
                current_batch_tasks = tasks_to_run[start_i:end_i]
                print(f"   🚀 [{os.path.basename(json_file)}] Processing batch [{chunk_idx+1}/{total_chunks}]...")

                pending_in_batch = current_batch_tasks
                MAX_RETRIES = 3
                
                while pending_in_batch:
                    batch_prompts = []
                    valid_tasks = []
                    
                    for task in pending_in_batch:
                        # Strict counterfactual logic evaluation Prompt
                        system_prompt = """You are a strict, adversarial Image Quality Assurance Judge. Your primary job is to FIND FLAWS and penalize AI-generated images that fail to strictly follow counterfactual physics or logic.
                        WARNING: AI models often generate normal objects instead of the requested counterfactual ones. Do NOT hallucinate success. Look closely for normal physics, normal shapes, or background inconsistencies.

                        You MUST output ONLY a valid JSON object. 
                        CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
                        Example format:
                        {
                        "1": {
                            "reasoning": "I observe that while the main object is altered, the shadow is normal. Also, there are standard spherical bubbles on the right side, which violates the flat requirement.",
                            "score": 0.0
                        }
                        }"""

                        user_content = f"""Please evaluate the image strictly against the following criteria. 
                        For each question, actively look for visual evidence that the image FAILS the criteria. If there is any ambiguity, normal physics, or partial failure, score it harshly (0.5 or 0.0).

                        Questions and Criteria:
                        {task['base_q_text']}"""

                        if task['retry_count'] > 0: 
                            user_content += "\n\n[ERROR] Previous response invalid. OUTPUT JSON ONLY."
                        
                        messages = [
                            {"role": "system", "content": system_prompt}, 
                            {"role": "user", "content": [{"type": "image", "image": task['image_obj']}, {"type": "text", "text": user_content}]}
                        ]
                        
                        try:
                            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            batch_prompts.append({"prompt": prompt_text, "multi_modal_data": {"image": task['image_obj']}})
                            valid_tasks.append(task)
                        except: 
                            pass

                    if not batch_prompts: break
                    outputs = llm.generate(batch_prompts, sampling_params)
                    next_round_pending = []
                    
                    for i, output in enumerate(outputs):
                        task = valid_tasks[i]
                        parsed = extract_json_robust(output.outputs[0].text)
                        
                        if parsed:
                            for temp_id, original_q in task['temp_id_map'].items():
                                res = parsed.get(temp_id, {})
                                if not res: res = parsed.get(f"Question ID {temp_id}", {})
                                
                                if isinstance(res, str):
                                    try: res = json.loads(res)
                                    except: res = {"reasoning": res, "score": 0.0}
                                
                                if not isinstance(res, dict): res = {}
                                
                                raw_score = res.get("score", 0.0)
                                final_score = 0.0
                                final_reasoning = res.get("reasoning", "Parsed")
                                
                                if isinstance(raw_score, dict):
                                    final_score = float(raw_score.get("score", 0.0))
                                    if "reasoning" in raw_score:
                                        final_reasoning = raw_score.get("reasoning")
                                else:
                                    try: final_score = float(raw_score)
                                    except: final_score = 0.0

                                final_results_list.append({
                                    "id": original_q.get("id"),
                                    "source_id": task['source_id'],
                                    "prompt_level": "l2",
                                    "image_filename": task['filename'],
                                    "question": original_q['question'],
                                    "weight": original_q.get('weight', 1),
                                    "score": final_score,
                                    "reasoning": final_reasoning,
                                    "evaluator": "Qwen3-VL-235B"
                                })
                        else:
                            if task['retry_count'] < MAX_RETRIES:
                                task['retry_count'] += 1
                                next_round_pending.append(task)
                            else:
                                for temp_id, original_q in task['temp_id_map'].items():
                                    final_results_list.append({"id": original_q.get("id"), "source_id": task['source_id'], "prompt_level": "l2", "score": 0.0, "weight": original_q.get('weight', 1), "reasoning": "FAILED"})
                    
                    pending_in_batch = next_round_pending
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_results_list, f, indent=2, ensure_ascii=False)

            if final_results_list:
                file_stats, image_scores = calculate_l2_stats(final_results_list, os.path.basename(json_file))
                all_files_statistics_model.append(file_stats)
                all_model_scores.extend(image_scores)
                print(f"📈 [Single File] L2 Average Score: {file_stats['l2_overall']}")

        # Calculate total score for the current model
        print(f"\n🌍 Calculating L2 total statistics for model [{current_model}]...")
        if all_model_scores:
            model_avg = sum(all_model_scores) / len(all_model_scores)
            model_summary = {"model_name": current_model, "l2_overall": round(model_avg, 4)}
            all_files_statistics_model.append(model_summary)
            global_comparison_stats.append(model_summary)
            print(f"   🔹 {current_model} L2 Total Score: {model_summary['l2_overall']}")
        else:
            print(f"⚠️ Model {current_model} did not generate any valid data.")

        # Save statistical summary for the current model
        model_summary_path = os.path.join(model_output_dir, "summary_l2_stats.json")
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_files_statistics_model, f, indent=2, ensure_ascii=False)

    # ================= 🏆 Generate Global Comparison Summary =================
    if global_comparison_stats:
        print("\n" + "="*80)
        print("🏆 All models processed! Generating global comparison summary table...")
        
        global_summary_path = os.path.join(OUTPUT_BASE_DIR, "global_l2_comparison.json")
        global_comparison_stats.sort(key=lambda x: x.get('l2_overall', 0.0), reverse=True)
        
        with open(global_summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_comparison_stats, f, indent=2, ensure_ascii=False)
            
        print(f"🎉 Global summary saved to: {global_summary_path}")
        print("Leaderboard Preview:")
        for rank, stat in enumerate(global_comparison_stats):
            print(f"  {rank+1}. {stat['model_name']:<25} | L2 Overall: {stat['l2_overall']:.4f}")

if __name__ == "__main__":
    main()
