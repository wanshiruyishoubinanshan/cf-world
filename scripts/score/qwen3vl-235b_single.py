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

# 1. 模型路径
MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8/snapshots/d464a056915e088a7621533813ed553ceea73a6e"

# 2. 路径配置
EVAL_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5"
IMAGE_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output/gpt-image-1.5"
OUTPUT_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score/gpt-image-1.5"

# 3. 硬件与推理配置
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 8192
BATCH_SIZE_PER_SAVE = 32

# 4. 评分阈值配置
L1_THRESHOLD = 0.5  # L1 图片得分低于此值，同组的 L2/L3 图片得分为 0

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

def get_image_dir_from_json_path(json_path):
    rel_path = os.path.relpath(json_path, EVAL_ROOT_DIR)
    path_obj = Path(rel_path)
    parent_dir = path_obj.parent
    stem = path_obj.stem.replace("_gemini", "").replace("_eval", "")
    target_image_dir = os.path.join(IMAGE_ROOT_DIR, parent_dir, stem)
    return target_image_dir, rel_path

def apply_l1_threshold_logic(results_list):
    """
    核心逻辑修正：
    1. 按 source_id 分组（每个 source_id 包含 L1, L2, L3 三张图）。
    2. 计算 L1 图片的加权平均分。
    3. 如果 L1 < 0.5，则将同组的 L2 和 L3 的所有题目分数设为 0。
    4. 计算每张图片（L1, L2, L3）的最终得分并写入 image_score 字段。
    """
    # 1. 按 source_id 分组
    grouped_data = {}
    for item in results_list:
        sid = str(item.get('source_id'))
        if sid not in grouped_data:
            grouped_data[sid] = []
        grouped_data[sid].append(item)
    
    # 2. 遍历每个 source_id (即每一组任务)
    for sid, items in grouped_data.items():
        # --- A. 计算 L1 图片的分数 ---
        l1_items = [x for x in items if str(x.get('prompt_level')).lower() == 'l1']
        
        l1_image_score = 0.0
        if l1_items:
            w_sum = sum(float(x.get('score', 0)) * float(x.get('weight', 1)) for x in l1_items)
            total_w = sum(float(x.get('weight', 1)) for x in l1_items)
            if total_w > 0:
                l1_image_score = w_sum / total_w
        
        # --- B. 阈值判定与熔断 ---
        is_l1_failed = l1_image_score < L1_THRESHOLD
        
        if is_l1_failed:
            # 如果 L1 挂了，找到同组的 L2 和 L3 题目，强制归零
            for item in items:
                level = str(item.get('prompt_level')).lower()
                if level in ['l2', 'l3']:
                    item['score'] = 0.0
                    # 追加说明
                    if "L1_Cutoff" not in str(item.get('reasoning', '')):
                        item['reasoning'] = str(item.get('reasoning', '')) + f" [System: Zeroed because L1 image score ({l1_image_score:.2f}) < {L1_THRESHOLD}]"

        # --- C. 计算并记录每张图片(L1, L2, L3)的最终得分 ---
        for level in ['l1', 'l2', 'l3']:
            level_items = [x for x in items if str(x.get('prompt_level')).lower() == level]
            if not level_items: continue
            
            w_sum = sum(float(x.get('score', 0)) * float(x.get('weight', 1)) for x in level_items)
            total_w = sum(float(x.get('weight', 1)) for x in level_items)
            img_score = w_sum / total_w if total_w > 0 else 0.0
            
            for x in level_items:
                x['image_score'] = round(img_score, 4)
                x['l1_ref_score'] = round(l1_image_score, 4)

    return results_list

def calculate_stats_from_results(results_list, filename):
    stats = {'l1': [], 'l2': [], 'l3': [], 'overall': []}
    unique_images = {}
    
    for item in results_list:
        sid = str(item.get('source_id'))
        level = str(item.get('prompt_level')).lower()
        score = float(item.get('image_score', 0.0))
        
        unique_key = f"{sid}_{level}"
        unique_images[unique_key] = {
            'level': level,
            'score': score
        }
        
    for key, data in unique_images.items():
        lvl = data['level']
        score = data['score']
        if lvl in stats:
            stats[lvl].append(score)
        stats['overall'].append(score)
            
    output = {"filename": filename}
    for key in ['l1', 'l2', 'l3', 'overall']:
        scores = stats.get(key, [])
        if scores:
            avg = sum(scores) / len(scores)
            output[key] = round(avg, 4)
        else:
            output[key] = 0.0
            
    return output, unique_images

# ================= 🚀 主逻辑 =================

def main():
    print(f"🔍 正在扫描评测文件: {EVAL_ROOT_DIR}/**/*.json")
    json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
    if not json_files:
        print("❌ 未找到任何 JSON 文件。")
        return
    
    all_unique_images_global = [] 
    
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
    
    all_files_statistics = []

    for json_idx, json_file in enumerate(json_files):
        print(f"\n" + "="*60)
        print(f"📂 处理文件 [{json_idx+1}/{len(json_files)}]: {os.path.basename(json_file)}")
        
        image_dir, rel_path = get_image_dir_from_json_path(json_file)
        output_file = os.path.join(OUTPUT_ROOT_DIR, rel_path.replace(".json", "_qwen235b.json"))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        final_results_list = []
        processed_keys = set()

        # A. 加载进度
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    final_results_list = json.load(f)
                for item in final_results_list:
                    k = (str(item.get('source_id')), str(item.get('prompt_level')))
                    processed_keys.add(k)
                print(f"⏩ 已加载现有进度: {len(processed_keys)} 张图片")
            except Exception:
                final_results_list = []

        # B. 准备任务
        if not os.path.exists(image_dir):
            print(f"⚠️  图片目录不存在: {image_dir} (跳过)")
            continue

        with open(json_file, 'r') as f:
            eval_data = json.load(f)

        image_tasks_map = {}
        for item in eval_data:
            key = (str(item['source_id']), str(item['prompt_level']))
            if key not in image_tasks_map: image_tasks_map[key] = []
            image_tasks_map[key].append(item)

        # ================= 🔥 核心修改：自动克隆 L2 任务作为 L3 任务 =================
        # 找出所有 L2 的 source_id
        l2_keys = [k for k in image_tasks_map.keys() if str(k[1]).lower() == 'l2']
        
        for sid, level in l2_keys:
            l3_key = (sid, 'l3')
            # 如果原本就没有 L3 的数据，我们就从 L2 克隆一份
            if l3_key not in image_tasks_map:
                l3_tasks = []
                for q in image_tasks_map[(sid, level)]:
                    q_clone = q.copy()
                    q_clone['prompt_level'] = 'l3'  # 强制标记为 l3
                    # 注意：这里我们直接 copy，所以 weight 等其他字段完全原样继承
                    l3_tasks.append(q_clone)
                image_tasks_map[l3_key] = l3_tasks
        # =========================================================================

        tasks_to_run = []
        for (sid, level), questions in image_tasks_map.items():
            if (sid, level) in processed_keys: continue
            
            img_name = f"{sid}_{level}.png"
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                if os.path.exists(img_path.replace(".png", ".jpg")):
                    img_path = img_path.replace(".png", ".jpg")
                else:
                    # 缺失图片处理（L3 图片缺失也会走到这里，记为 0 分）
                    for q in questions:
                        res = q.copy()
                        res.update({"score": 0.0, "reasoning": "Image missing", "evaluator": "None"})
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
                    "level": level,
                    "filename": img_name,
                    "image_obj": image_obj,
                    "temp_id_map": temp_id_map,
                    "base_q_text": q_text,
                    "retry_count": 0
                })
            except Exception as e:
                print(f"   ❌ 图片读取错误 {img_name}: {e}")

        # C. 推理逻辑
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
                    # ================= 🔥 动态 Prompt 逻辑 =================
                    if task['level'].lower() == 'l1':
                        # L1: 常规、平衡的物理常识与事实评测
                        system_prompt = """You are an objective and balanced Image Quality Assurance Assistant. Your job is to evaluate whether an AI-generated image accurately reflects standard real-world physics, common sense, and the provided factual criteria.
                        Note: Minor AI artifacts or slight imperfections are acceptable as long as the main subject and overall scene align with the prompt and basic reality. Do not be overly harsh on negligible details.

                        You MUST output ONLY a valid JSON object. 
                        CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
                        Example format:
                        {
                        "1": {
                            "reasoning": "The image clearly shows a normal apple on a table. The lighting and shadows are realistic. There are some minor blurry pixels in the background, but it does not affect the overall factual accuracy.",
                            "score": 1.0
                        }
                        }"""

                        user_content = f"""Please evaluate the image based on the following criteria. 
                        Assess whether the image generally satisfies the factual requirements and common sense. If the criteria are mostly met despite minor AI flaws, you can give a high score (e.g., 1.0). If it partially fails or has noticeable logical errors, score it accordingly (e.g., 0.5). Only give 0.0 if it completely fails the criteria.

                        Questions and Criteria:
                        {task['base_q_text']}"""

                    else:
                        # L2 & L3: 严格的、对抗性的反事实逻辑评测 (保留您原有的 Prompt)
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
                        For each question, actively look for visual evidence that the image FAILS the criteria. If there is any ambiguity, normal physics, or partial failure, score it harshly.

                        Questions and Criteria:
                        {task['base_q_text']}"""
                    # =========================================================

                    # 处理重试逻辑时的附加提示
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
                            
                            # ================= 🔥 新增的安全校验 =================
                            # 1. 如果模型返回的是字符串（可能是嵌套的 JSON 字符串，也可能是纯文本废话）
                            if isinstance(res, str):
                                try:
                                    res = json.loads(res)
                                except json.JSONDecodeError:
                                    # 如果解析失败，说明是纯文本，将其作为 reasoning，分数为 0
                                    res = {"reasoning": res, "score": 0.0}
                            
                            # 2. 最后的兜底：确保 res 绝对是一个字典
                            if not isinstance(res, dict):
                                res = {}
                            # =====================================================
                            
                            raw_score = res.get("score", 0.0)
                            final_score = 0.0
                            final_reasoning = res.get("reasoning", "Parsed")
                            
                            if isinstance(raw_score, dict):
                                final_score = float(raw_score.get("score", 0.0))
                                if "reasoning" in raw_score:
                                    final_reasoning = raw_score.get("reasoning")
                            else:
                                try:
                                    final_score = float(raw_score)
                                except (ValueError, TypeError):
                                    final_score = 0.0

                            final_results_list.append({
                                "id": original_q.get("id"),
                                "source_id": task['source_id'],
                                "prompt_level": task['level'],
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
                                final_results_list.append({"id": original_q.get("id"), "prompt_level": task['level'], "score": 0.0, "weight": original_q.get('weight', 1), "reasoning": "FAILED"})
                pending_in_batch = next_round_pending
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)

        # D. 统计当前文件
        if final_results_list:
            print("   ⚖️  正在应用 L1 阈值逻辑并计算图片得分...")
            
            final_results_list = apply_l1_threshold_logic(final_results_list)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)
            
            file_stats, unique_images_map = calculate_stats_from_results(final_results_list, os.path.basename(json_file))
            all_files_statistics.append(file_stats)
            
            print(f"📈 [单文件] Overall: {file_stats['overall']} (L1:{file_stats.get('l1')} L2:{file_stats.get('l2')} L3:{file_stats.get('l3')})")

            all_unique_images_global.extend(unique_images_map.values())

    # --- 🔥 最后一步：计算全局总分 ---
    print("\n" + "="*60)
    print("🌍 计算全局统计 (Global Statistics)...")
    
    if all_unique_images_global:
        global_stats = {'l1': [], 'l2': [], 'l3': [], 'overall': []}
        for img_data in all_unique_images_global:
            lvl = img_data['level']
            score = img_data['score']
            if lvl in global_stats: global_stats[lvl].append(score)
            global_stats['overall'].append(score)
            
        final_summary = {"filename": "TOTAL_AVERAGE"}
        for key in ['l1', 'l2', 'l3', 'overall']:
            scores = global_stats.get(key, [])
            if scores:
                final_summary[key] = round(sum(scores) / len(scores), 4)
            else:
                final_summary[key] = 0.0
        
        all_files_statistics.append(final_summary)
        
        print(f"   🔹 TOTAL L1: {final_summary['l1']}")
        print(f"   🔹 TOTAL L2: {final_summary['l2']}")
        print(f"   🔹 TOTAL L3: {final_summary['l3']}")
        print(f"   🔹 TOTAL Overall: {final_summary['overall']}")
    else:
        print("⚠️ 没有产生任何有效数据。")

    summary_path = os.path.join(OUTPUT_ROOT_DIR, "summary_weighted_stats.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_files_statistics, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 所有任务完成！统计汇总已保存至: {summary_path}")

if __name__ == "__main__":
    main()
