import os
import json
import time
import base64
import glob
import re
import httpx
import traceback
from io import BytesIO
from pathlib import Path
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
import argparse  # New: For receiving command line arguments

try:
    from PIL import Image
except ImportError:
    Image = None

# ================= ⚙️ Configuration Area =================

# 1. API Configuration
# Replace with your actual API endpoint and key
BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "sk-your-api-key-here"
MODEL_NAME = "gemini-3-pro-preview"

print(f"Using API Base URL: {BASE_URL}")

# 2. Path Configuration (Anonymized)
EVAL_ROOT_DIR = "./data/counterfactual/eval_question"
OUTPUT_BASE_DIR = "./data/counterfactual/output"
SCORE_BASE_DIR = "./data/counterfactual/score"

# ================= 🛠️ Utility Functions =================

def init_client():
    http_client = httpx.Client(timeout=120.0, verify=False)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

def encode_image(image_path, max_side: int = 1024, quality: int = 85):
    try:
        if Image is not None:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                scale = min(1.0, max_side / max(w, h))
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                img_bytes = buf.getvalue()
        else:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to read image {image_path}: {e}")
        return None

def extract_json_robust(text):
    if not text: return None
    text = re.sub(r'^```(json)?', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def get_image_dir_from_json_path(json_path, current_image_root):
    rel_path = os.path.relpath(json_path, EVAL_ROOT_DIR)
    path_obj = Path(rel_path)
    parent_dir = path_obj.parent
    stem = path_obj.stem.replace("_gemini", "").replace("_eval", "")
    target_image_dir = os.path.join(current_image_root, parent_dir, stem)
    return target_image_dir, rel_path

# --- Statistical Logic ---

def calculate_image_score(batch_results):
    total_score = 0.0
    total_weight = 0.0
    for item in batch_results:
        score = float(item.get('score', 0.0))
        weight = float(item.get('weight', 1.0))
        total_score += score * weight
        total_weight += weight
    return total_score / total_weight if total_weight > 0 else 0.0

def process_image_scores(results_list):
    images_map = {}
    for item in results_list:
        sid = str(item.get('source_id'))
        level = str(item.get('prompt_level', 'unknown')).lower()
        score = float(item.get('score', 0.0))
        weight = float(item.get('weight', 1.0))
        
        img_key = f"{sid}_{level}"
        if img_key not in images_map:
            images_map[img_key] = {'level': level, 'weighted_score_sum': 0.0, 'total_weight': 0.0}
        
        images_map[img_key]['weighted_score_sum'] += score * weight
        images_map[img_key]['total_weight'] += weight

    processed_images = []
    for key, data in images_map.items():
        final_img_score = data['weighted_score_sum'] / data['total_weight'] if data['total_weight'] > 0 else 0.0
        processed_images.append({'level': data['level'], 'score': final_img_score})
        
    return processed_images

def calculate_stats_from_images(image_list, filename):
    stats = {'l1': [], 'l2': [], 'l3': [], 'overall': []}
    for img in image_list:
        lvl = img['level']
        score = img['score']
        if lvl in stats: stats[lvl].append(score)
        stats['overall'].append(score)
        
    output = {"filename": filename}
    for key in ['l1', 'l2', 'l3', 'overall']:
        scores = stats.get(key, [])
        output[key] = round(sum(scores) / len(scores), 4) if scores else 0.0
    return output

# ================= 🧠 Core Logic =================

# Lenient Version
def process_single_image_batch(client, image_path, questions_list, level):
    base64_image = encode_image(image_path)
    if not base64_image: return None

    base_q_text = ""
    temp_id_map = {}
    for idx, q in enumerate(questions_list):
        temp_id = str(idx + 1)
        temp_id_map[temp_id] = q
        base_q_text += f'\nQuestion ID "{temp_id}":\nQuestion: {q["question"]}\nCriteria: {q["evaluation_criteria"]}\n---'

    if level.lower() == 'l1':
        # --- L1: Highly lenient basic factual and common sense evaluation ---
        system_prompt = """You are an Image Quality Assurance Assistant. Your job is to evaluate whether an AI-generated image generally captures the main idea of the provided factual criteria and common sense.

You MUST output ONLY a valid JSON object. 
CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
Example format:
{
"1": {
    "reasoning": "The image clearly shows an apple on a table. Although the table's texture is a bit warped and the lighting is slightly unnatural, the core factual requirement is fully met.",
    "score": 1.0
}
}"""
        user_content_text = f"""Please evaluate the image based on the following criteria. 
If the image successfully conveys the core concept requested, give it a 1.0. Deduct points (e.g., 0.5 to 0.8) only if there are significant missing elements or major deviations from the prompt. Give 0.0 only if the image is completely unrelated to the criteria.

Questions and Criteria:
{base_q_text}"""
    else:
        # --- L2/L3: Rational counterfactual evaluation with flexible scoring ---
        system_prompt = """You are an analytical Image Quality Assurance Evaluator. Your primary job is to verify if the AI successfully generated the requested counterfactual or illogical elements.
WARNING: AI models often default to normal objects instead of the requested counterfactual ones. Please check the main subject carefully to ensure it breaks normal physics as requested.
Note: Focus on whether the core counterfactual instruction is met. Minor AI artifacts, slight edge blurriness, or imperfect backgrounds are acceptable. If the main counterfactual goal is clearly achieved despite minor visual flaws, score it between 0.5 and 1.0 depending on the severity of the flaws. Score 0.0 only if it completely fails the counterfactual instruction or reverts to normal physics.

You MUST output ONLY a valid JSON object. 
CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
Example format:
{
"1": {
    "reasoning": "The image successfully shows a square apple, meeting the core counterfactual requirement. There are some minor AI artifacts in the background shadow, but they do not detract from the main instruction.",
    "score": 0.8
}
}"""
        user_content_text = f"""Please evaluate the image against the following criteria. 
Focus primarily on whether the counterfactual requirements are met. Do not be overly harsh on standard AI generation flaws (like slight blurriness or weird background objects) as long as the main subject successfully follows the illogical/counterfactual prompt.

Questions and Criteria:
{base_q_text}"""

    MAX_RETRIES = 100
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_content_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=5500
            )
            content = response.choices[0].message.content

            print(f"\n[Debug Raw Output] Attempt {attempt+1}:")
            print(content)
            print("-" * 40)

            parsed_result = extract_json_robust(content)
            
            if not parsed_result:
                print(f"      ⚠️ Parsing empty, retrying ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(5)
                continue 
                
            batch_results = []
            for temp_id, original_q in temp_id_map.items():
                res = parsed_result.get(temp_id, {})
                if not res: res = parsed_result.get(f"Question ID {temp_id}", {})
                
                batch_results.append({
                    "id": original_q.get("id"),
                    "source_id": original_q.get("source_id"),
                    "prompt_level": original_q.get("prompt_level"),
                    "question": original_q.get("question"),
                    "weight": original_q.get("weight", 1),
                    "score": float(res.get("score", 0.0)),
                    "reasoning": res.get("reasoning", "Parsed"),
                    "scoring_model": MODEL_NAME
                })
            return batch_results

        except Exception as e:
            wait_time = 15
            print(f"      ⚠️ Error occurred ({e.__class__.__name__}: {e}), waiting {wait_time}s before retrying... ({attempt+1}/{MAX_RETRIES})")
            time.sleep(wait_time)

    print("      ❌ Reached maximum retries, abandoning this image.")
    return None


# ================= 🚀 Main Program =================

def main():
    # ================= New: Parse command line arguments =================
    parser = argparse.ArgumentParser(description="Run evaluation for a specific model")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Specify a single model name to process (e.g., --model sd3)"
    )
    args = parser.parse_args()
    # ========================================================

    client = init_client()
    
    print(f"🔍 Scanning evaluation files: {EVAL_ROOT_DIR}/**/*.json")
    json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
    if not json_files:
        print("❌ No JSON evaluation files found.")
        return

    if not os.path.exists(OUTPUT_BASE_DIR):
        print(f"❌ Base output directory does not exist: {OUTPUT_BASE_DIR}")
        return
        
    # ================= New: Support specific model =================
    all_models = [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]
    
    if args.model:
        # User specified a model, process only this one
        if args.model not in all_models:
            print(f"❌ Specified model does not exist! Available models: {all_models}")
            return
        models = [args.model]
        print(f"🚀 Processing only specified model: {args.model}")
    else:
        # Not specified, process all
        models = all_models
        print(f"📦 Found {len(models)} models to process: {models}")
    # ========================================================

    for model_name in models:
        print("\n" + "★"*60)
        print(f"🚀 Starting to process model: {model_name}")
        print("★"*60)
        
        current_image_root = os.path.join(OUTPUT_BASE_DIR, model_name)
        current_output_root = os.path.join(SCORE_BASE_DIR, f"gemini-{model_name}")
        
        all_files_statistics = []
        all_images_global = []

        for idx, json_file in enumerate(json_files):
            print(f"\n📂 [{model_name}] Processing file [{idx+1}/{len(json_files)}]: {os.path.basename(json_file)}")
            
            image_dir, rel_path = get_image_dir_from_json_path(json_file, current_image_root)
            output_file = os.path.join(current_output_root, rel_path.replace(".json", "_scores.json"))
            
            final_results = []
            processed_keys = set() 
            l1_score_map = {} 
            
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        final_results = json.load(f)
                        
                        temp_l1_results = {}
                        for item in final_results:
                            sid = str(item.get('source_id'))
                            lvl = str(item.get('prompt_level')).lower()
                            processed_keys.add((sid, lvl))
                            
                            if lvl == 'l1':
                                if sid not in temp_l1_results: temp_l1_results[sid] = []
                                temp_l1_results[sid].append(item)
                        
                        for sid, res_list in temp_l1_results.items():
                            l1_score_map[sid] = calculate_image_score(res_list)
                            
                    print(f"⏩ Loaded existing progress: {len(processed_keys)} images processed")
                except Exception as e:
                    print(f"⚠️ Error reading existing file ({e}), starting over.")
                    final_results = []

            if not os.path.exists(image_dir):
                print(f"⚠️ Image directory does not exist: {image_dir} (Skipping)")
                continue
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(json_file, 'r') as f:
                raw_data = json.load(f)
            
            tasks_by_source = {}
            for item in raw_data:
                sid = str(item['source_id'])
                lvl = str(item['prompt_level']).lower()
                if sid not in tasks_by_source:
                    tasks_by_source[sid] = {}
                if lvl not in tasks_by_source[sid]:
                    tasks_by_source[sid][lvl] = []
                tasks_by_source[sid][lvl].append(item)
            
            # Auto-clone L2 -> L3
            for sid, levels_dict in tasks_by_source.items():
                if 'l2' in levels_dict and 'l3' not in levels_dict:
                    l3_tasks = []
                    for q in levels_dict['l2']:
                        q_clone = q.copy()
                        q_clone['prompt_level'] = 'l3'
                        l3_tasks.append(q_clone)
                    levels_dict['l3'] = l3_tasks

            print(f"📊 Total tasks: {len(tasks_by_source)} Source IDs")
            
            for i, (sid, levels_dict) in enumerate(tasks_by_source.items()):
                sorted_levels = sorted(levels_dict.keys(), key=lambda x: {'l1': 1, 'l2': 2, 'l3': 3}.get(x, 99))
                
                for level in sorted_levels:
                    questions = levels_dict[level]
                    img_name = f"{sid}_{level}.png"
                    
                    if (sid, level) in processed_keys:
                        continue

                    img_path = os.path.join(image_dir, img_name)
                    
                    if not os.path.exists(img_path):
                        if os.path.exists(img_path.replace(".png", ".jpg")):
                            img_path = img_path.replace(".png", ".jpg")

                    print(f"   [{i+1}/{len(tasks_by_source)}] Processing: {os.path.basename(img_path)} ...", end="", flush=True)
                    
                    batch_res = []
                    
                    if level in ['l2', 'l3']:
                        l1_score = l1_score_map.get(sid, 0.0)
                        if l1_score < 0.5:
                            print(f" ⏭️ Skipped (L1 score is {l1_score:.2f} < 0.5, assigning 0)")
                            for q in questions:
                                res = q.copy()
                                res.update({
                                    "score": 0.0, 
                                    "reasoning": "Skipped due to L1 score < 0.5", 
                                    "scoring_model": "None"
                                })
                                batch_res.append(res)
                    
                    if not batch_res:
                        if not os.path.exists(img_path):
                            print(" ⚠️ Missing (0 score)")
                            for q in questions:
                                res = q.copy()
                                res.update({"score": 0.0, "reasoning": "Image missing", "scoring_model": "None"})
                                batch_res.append(res)
                        else:
                            start_t = time.time()
                            api_res = process_single_image_batch(client, img_path, questions, level)
                            dur = time.time() - start_t
                            
                            if api_res:
                                print(f" ✅ Completed ({dur:.2f}s)")
                                batch_res.extend(api_res)
                            else:
                                print(f" ❌ Failed (0 score)")
                                for q in questions:
                                    res = q.copy()
                                    res.update({"score": 0.0, "reasoning": "API Failed", "scoring_model": MODEL_NAME})
                                    batch_res.append(res)
                    
                    current_image_score = calculate_image_score(batch_res)
                    
                    if level == 'l1':
                        l1_score_map[sid] = current_image_score
                    
                    current_l1_score = l1_score_map.get(sid, 0.0)
                    
                    for item in batch_res:
                        item['image_score'] = round(current_image_score, 4)
                        item['l1_score'] = round(current_l1_score, 4)
                        
                    final_results.extend(batch_res)
                    processed_keys.add((sid, level))
                    
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(final_results, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"   ⚠️ Save failed: {e}")

                    time.sleep(0.5)

            if final_results:
                file_images = process_image_scores(final_results)
                file_stats = calculate_stats_from_images(file_images, os.path.basename(json_file))
                all_files_statistics.append(file_stats)
                all_images_global.extend(file_images)
                
                print(f"📈 [Statistics] Overall: {file_stats['overall']} (L1:{file_stats.get('l1')} L2:{file_stats.get('l2')} L3:{file_stats.get('l3')})")

        print(f"\n🌍 Calculating [{model_name}] Global Statistics...")
        if all_images_global:
            global_stats = calculate_stats_from_images(all_images_global, "TOTAL_AVERAGE")
            all_files_statistics.append(global_stats)
            print(f"   🔹 TOTAL L1: {global_stats['l1']}")
            print(f"   🔹 TOTAL L2: {global_stats['l2']}")
            print(f"   🔹 TOTAL L3: {global_stats['l3']}")
            print(f"   🔹 TOTAL Overall: {global_stats['overall']}")
        else:
            print("⚠️ No valid data generated.")

        summary_path = os.path.join(current_output_root, "summary_stats.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_files_statistics, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 Model {model_name} processing complete! Summary saved to: {summary_path}\n")

    print("🏁 All models have been processed!")

if __name__ == "__main__":
    main()
