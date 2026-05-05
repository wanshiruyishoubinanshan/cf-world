import os
import json
import re
import glob
import traceback
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= ⚙️ 配置区域 =================

MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8/snapshots/d464a056915e088a7621533813ed553ceea73a6e"

# 基础路径配置
# 指定要评测的单一 JSON 文件
EVAL_JSON_FILE = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5/physics/Classical Mechanics_gemini.json"
# 用于计算相对路径的基准目录
EVAL_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5"

IMAGE_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output/try_3_idea"
OUTPUT_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score/try_3_idea_score"

# 硬件与推理配置 (4张GPU完美适配)
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 8192
BATCH_SIZE_PER_SAVE = 32

# 图片变体后缀
IMAGE_VARIANTS = ["l1","l2","l3"]

# ================= 🛠️ 工具函数 =================

def extract_json_robust(text):
    """从大模型回复中提取 JSON"""
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

def calculate_image_scores(results_list):
    """计算每张图片的加权平均分，并写入 image_score 字段"""
    grouped = {}
    for item in results_list:
        key = f"{item['source_id']}_{item['image_variant']}"
        if key not in grouped: grouped[key] = []
        grouped[key].append(item)
        
    for key, items in grouped.items():
        w_sum = sum(float(x.get('score', 0)) * float(x.get('weight', 1)) for x in items)
        total_w = sum(float(x.get('weight', 1)) for x in items)
        img_score = w_sum / total_w if total_w > 0 else 0.0
        
        for x in items:
            x['image_score'] = round(img_score, 4)
            
    return results_list

def calculate_stats_from_results(results_list, filename):
    """按变体类型统计分数"""
    stats = {v: [] for v in IMAGE_VARIANTS}
    stats['overall'] = []
    unique_images = {}
    
    for item in results_list:
        sid = str(item.get('source_id'))
        variant = str(item.get('image_variant'))
        score = float(item.get('image_score', 0.0))
        
        unique_key = f"{sid}_{variant}"
        unique_images[unique_key] = {'variant': variant, 'score': score}
        
    for key, data in unique_images.items():
        var = data['variant']
        score = data['score']
        if var in stats:
            stats[var].append(score)
        stats['overall'].append(score)
            
    output = {"filename": filename}
    for key in IMAGE_VARIANTS + ['overall']:
        scores = stats.get(key, [])
        if scores:
            output[key] = round(sum(scores) / len(scores), 4)
        else:
            output[key] = 0.0
            
    return output, unique_images

# ================= 🚀 主逻辑 =================

def main():
    print(f"🔍 正在读取指定的评测文件: {EVAL_JSON_FILE}")
    
    if not os.path.exists(EVAL_JSON_FILE):
        print(f"❌ 未找到指定的 JSON 文件: {EVAL_JSON_FILE}")
        return
        
    # 直接将目标文件放入列表
    json_files = [EVAL_JSON_FILE]
    
    if not os.path.exists(IMAGE_BASE_DIR):
        print(f"❌ 图片基础目录不存在: {IMAGE_BASE_DIR}")
        return
        
    model_names = [d for d in os.listdir(IMAGE_BASE_DIR) if os.path.isdir(os.path.join(IMAGE_BASE_DIR, d))]
    model_names.sort()
    print(f"📦 发现 {len(model_names)} 个模型待评测: {', '.join(model_names)}")
    
    print(f"\n⏳ 正在加载 Qwen3-VL-235B (TP={TENSOR_PARALLEL_SIZE})...")
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
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=2048, stop_token_ids=[151645, 151643])
    global_comparison_stats = []

    for model_idx, current_model in enumerate(model_names):
        print("\n" + "★"*80)
        print(f"🚀 开始处理模型 [{model_idx+1}/{len(model_names)}]: {current_model}")
        print("★"*80)
        
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, f"qwen_{current_model}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 当前模型的图片目录 (扁平化，直接在模型文件夹下)
        current_model_image_dir = os.path.join(IMAGE_BASE_DIR, current_model)
        
        all_unique_images_model = [] 
        all_files_statistics_model = []

        for json_idx, json_file in enumerate(json_files):
            print(f"\n📂 [{current_model}] 处理文件: {os.path.basename(json_file)}")
            
            # 计算相对路径以便在 output 目录中保持结构
            rel_path = os.path.relpath(json_file, EVAL_BASE_DIR)
            output_file = os.path.join(model_output_dir, rel_path.replace(".json", "_qwen235b.json"))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            final_results_list = []
            processed_keys = set()

            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        final_results_list = json.load(f)
                    for item in final_results_list:
                        k = (str(item.get('source_id')), str(item.get('image_variant')))
                        processed_keys.add(k)
                    print(f"⏩ 已加载现有进度: {len(processed_keys)} 张图片")
                except Exception:
                    final_results_list = []

            with open(json_file, 'r') as f:
                eval_data = json.load(f)

            # 仅提取 L2 问题
            l2_tasks_map = {}
            for item in eval_data:
                if str(item.get('prompt_level', '')).lower() == 'l2':
                    sid = str(item['source_id'])
                    if sid not in l2_tasks_map: l2_tasks_map[sid] = []
                    l2_tasks_map[sid].append(item)

            tasks_to_run = []
            for sid, questions in l2_tasks_map.items():
                # 为每个 source_id 寻找三个变体的图片
                for variant in IMAGE_VARIANTS:
                    if (sid, variant) in processed_keys: continue
                    
                    img_name = f"{sid}_{variant}.png"
                    img_path = os.path.join(current_model_image_dir, img_name)
                    
                    if not os.path.exists(img_path):
                        if os.path.exists(img_path.replace(".png", ".jpg")):
                            img_path = img_path.replace(".png", ".jpg")
                        else:
                            # 缺失图片处理
                            for q in questions:
                                res = q.copy()
                                res.update({
                                    "image_variant": variant,
                                    "image_filename": img_name,
                                    "score": 0.0, 
                                    "reasoning": "Image missing", 
                                    "evaluator": "None"
                                })
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
                            "variant": variant,
                            "filename": img_name,
                            "image_obj": image_obj,
                            "temp_id_map": temp_id_map,
                            "base_q_text": q_text,
                            "retry_count": 0
                        })
                    except Exception as e:
                        print(f"   ❌ 图片读取错误 {img_name}: {e}")

            chunk_size = BATCH_SIZE_PER_SAVE
            total_chunks = (len(tasks_to_run) + chunk_size - 1) // chunk_size

            for chunk_idx in range(total_chunks):
                start_i = chunk_idx * chunk_size
                end_i = min((chunk_idx + 1) * chunk_size, len(tasks_to_run))
                current_batch_tasks = tasks_to_run[start_i:end_i]
                print(f"   🚀 处理批次 [{chunk_idx+1}/{total_chunks}]...")

                pending_in_batch = current_batch_tasks
                MAX_RETRIES = 3
                while pending_in_batch:
                    batch_prompts = []
                    valid_tasks = []
                    for task in pending_in_batch:
                        # 统一使用严格的对抗性 Prompt
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
                                    "image_variant": task['variant'],
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
                                    final_results_list.append({
                                        "id": original_q.get("id"), 
                                        "source_id": task['source_id'],
                                        "image_variant": task['variant'],
                                        "score": 0.0, 
                                        "weight": original_q.get('weight', 1), 
                                        "reasoning": "FAILED"
                                    })
                    pending_in_batch = next_round_pending
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_results_list, f, indent=2, ensure_ascii=False)

            if final_results_list:
                # 计算每张图片的加权得分
                final_results_list = calculate_image_scores(final_results_list)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_results_list, f, indent=2, ensure_ascii=False)
                
                file_stats, unique_images_map = calculate_stats_from_results(final_results_list, os.path.basename(json_file))
                all_files_statistics_model.append(file_stats)
                all_unique_images_model.extend(unique_images_map.values())
                
                print(f"📈 [单文件] Overall: {file_stats['overall']} (C1:{file_stats.get('')} C2:{file_stats.get('C2_Fictional')} C3:{file_stats.get('C3_Visual')})")

        print(f"\n🌍 计算模型 [{current_model}] 的总统计...")
        if all_unique_images_model:
            model_stats = {v: [] for v in IMAGE_VARIANTS}
            model_stats['overall'] = []
            
            for img_data in all_unique_images_model:
                var = img_data['variant']
                score = img_data['score']
                if var in model_stats: model_stats[var].append(score)
                model_stats['overall'].append(score)
                
            model_summary = {"filename": "TOTAL_AVERAGE", "model_name": current_model}
            for key in IMAGE_VARIANTS + ['overall']:
                scores = model_stats.get(key, [])
                model_summary[key] = round(sum(scores) / len(scores), 4) if scores else 0.0
            
            all_files_statistics_model.append(model_summary)
            global_comparison_stats.append(model_summary)
            
            print(f"   🔹 {current_model} C1_Common: {model_summary['C1_Common']}")
            print(f"   🔹 {current_model} C2_Fictional: {model_summary['C2_Fictional']}")
            print(f"   🔹 {current_model} C3_Visual: {model_summary['C3_Visual']}")
            print(f"   🔹 {current_model} Overall: {model_summary['overall']}")
        else:
            print(f"⚠️ 模型 {current_model} 没有产生任何有效数据。")

        model_summary_path = os.path.join(model_output_dir, "summary_weighted_stats.json")
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_files_statistics_model, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("🏆 所有模型处理完毕！正在生成全局对比汇总表...")
    
    global_summary_path = os.path.join(OUTPUT_BASE_DIR, "global_models_comparison.json")
    global_comparison_stats.sort(key=lambda x: x.get('overall', 0.0), reverse=True)
    
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        json.dump(global_comparison_stats, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 全局汇总已保存至: {global_summary_path}")
    print("排行榜预览:")
    for rank, stat in enumerate(global_comparison_stats):
        print(f"  {rank+1}. {stat['model_name']:<25} | Overall: {stat['overall']:.4f} | C1: {stat['C1_Common']:.4f} | C2: {stat['C2_Fictional']:.4f} | C3: {stat['C3_Visual']:.4f}")

if __name__ == "__main__":
    main()
