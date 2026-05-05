# import os
# import json
# import time
# import base64
# import glob
# import re
# import httpx
# import traceback
# from io import BytesIO
# from pathlib import Path
# from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

# try:
#     from PIL import Image
# except ImportError:
#     Image = None

# # ================= ⚙️ 配置区域 =================

# # 1. API 配置
# # BASE_URL = "http://35.220.164.252:3888/v1"
# BASE_URL = "https://api.whatai.cc/v1"
# # API_KEY = "sk-aEgLZS952k0LiNx7kJYJOo5tkhOERlNQfxhnV5xIugxAzltm"
# API_KEY = "sk-o6mBpaewBQ1TvcoQETr2mwpl6kcLIG4j59QQymhnwdNAUOtw"
# MODEL_NAME = "gemini-3-pro-preview" 

# # 2. 路径配置 (修改为动态遍历的基础路径)
# EVAL_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5"
# OUTPUT_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output"
# SCORE_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score"

# # ================= 🛠️ 工具函数 =================

# def init_client():
#     http_client = httpx.Client(timeout=120.0, verify=False)
#     return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# def encode_image(image_path, max_side: int = 1024, quality: int = 85):
#     try:
#         if Image is not None:
#             with Image.open(image_path) as img:
#                 img = img.convert("RGB")
#                 w, h = img.size
#                 scale = min(1.0, max_side / max(w, h))
#                 if scale < 1.0:
#                     new_size = (int(w * scale), int(h * scale))
#                     img = img.resize(new_size, Image.LANCZOS)
#                 buf = BytesIO()
#                 img.save(buf, format="JPEG", quality=quality)
#                 img_bytes = buf.getvalue()
#         else:
#             with open(image_path, "rb") as f:
#                 img_bytes = f.read()
#         return base64.b64encode(img_bytes).decode("utf-8")
#     except Exception as e:
#         print(f"❌ 读取图片失败 {image_path}: {e}")
#         return None

# def extract_json_robust(text):
#     if not text: return None
#     text = re.sub(r'^```(json)?', '', text, flags=re.MULTILINE)
#     text = re.sub(r'```$', '', text, flags=re.MULTILINE)
#     match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
#     if match:
#         try: return json.loads(match.group(0))
#         except: pass
#     return None

# def get_image_dir_from_json_path(json_path, current_image_root):
#     rel_path = os.path.relpath(json_path, EVAL_ROOT_DIR)
#     path_obj = Path(rel_path)
#     parent_dir = path_obj.parent
#     stem = path_obj.stem.replace("_gemini", "").replace("_eval", "")
#     target_image_dir = os.path.join(current_image_root, parent_dir, stem)
#     return target_image_dir, rel_path

# # --- 统计逻辑 ---

# def calculate_image_score(batch_results):
#     """计算单张图片(一个batch)的加权平均分"""
#     total_score = 0.0
#     total_weight = 0.0
#     for item in batch_results:
#         score = float(item.get('score', 0.0))
#         weight = float(item.get('weight', 1.0))
#         total_score += score * weight
#         total_weight += weight
#     return total_score / total_weight if total_weight > 0 else 0.0

# def process_image_scores(results_list):
#     """将所有题目列表聚合为图片得分列表 (用于最终统计)"""
#     images_map = {}
#     for item in results_list:
#         sid = str(item.get('source_id'))
#         level = str(item.get('prompt_level', 'unknown')).lower()
#         score = float(item.get('score', 0.0))
#         weight = float(item.get('weight', 1.0))
        
#         img_key = f"{sid}_{level}"
#         if img_key not in images_map:
#             images_map[img_key] = {'level': level, 'weighted_score_sum': 0.0, 'total_weight': 0.0}
        
#         images_map[img_key]['weighted_score_sum'] += score * weight
#         images_map[img_key]['total_weight'] += weight

#     processed_images = []
#     for key, data in images_map.items():
#         final_img_score = data['weighted_score_sum'] / data['total_weight'] if data['total_weight'] > 0 else 0.0
#         processed_images.append({'level': data['level'], 'score': final_img_score})
        
#     return processed_images

# def calculate_stats_from_images(image_list, filename):
#     stats = {'l1': [], 'l2': [], 'l3': [], 'overall': []}
#     for img in image_list:
#         lvl = img['level']
#         score = img['score']
#         if lvl in stats: stats[lvl].append(score)
#         stats['overall'].append(score)
        
#     output = {"filename": filename}
#     for key in ['l1', 'l2', 'l3', 'overall']:
#         scores = stats.get(key, [])
#         output[key] = round(sum(scores) / len(scores), 4) if scores else 0.0
#     return output

# # ================= 🧠 核心逻辑 =================

# def process_single_image_batch(client, image_path, questions_list, level):
#     base64_image = encode_image(image_path)
#     if not base64_image: return None

#     base_q_text = ""
#     temp_id_map = {}
#     for idx, q in enumerate(questions_list):
#         temp_id = str(idx + 1)
#         temp_id_map[temp_id] = q
#         base_q_text += f'\nQuestion ID "{temp_id}":\nQuestion: {q["question"]}\nCriteria: {q["evaluation_criteria"]}\n---'

#     if level.lower() == 'l1':
#         system_prompt = """You are an objective and balanced Image Quality Assurance Assistant. Your job is to evaluate whether an AI-generated image accurately reflects standard real-world physics, common sense, and the provided factual criteria.
# Note: Minor AI artifacts or slight imperfections are acceptable as long as the main subject and overall scene align with the prompt and basic reality. Do not be overly harsh on negligible details.

# You MUST output ONLY a valid JSON object. 
# CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
# Example format:
# {
# "1": {
#     "reasoning": "The image clearly shows a normal apple on a table. The lighting and shadows are realistic. There are some minor blurry pixels in the background, but it does not affect the overall factual accuracy.",
#     "score": 1.0
# }
# }"""
#         user_content_text = f"""Please evaluate the image based on the following criteria. 
# Assess whether the image generally satisfies the factual requirements and common sense. If the criteria are mostly met despite minor AI flaws, you can give a high score (e.g., 1.0). If it partially fails or has noticeable logical errors, score it accordingly (e.g., 0.5). Only give 0.0 if it completely fails the criteria.

# Questions and Criteria:
# {base_q_text}"""
#     else:
#         system_prompt = """You are a strict, adversarial Image Quality Assurance Judge. Your primary job is to FIND FLAWS and penalize AI-generated images that fail to strictly follow counterfactual physics or logic.
# WARNING: AI models often generate normal objects instead of the requested counterfactual ones. Do NOT hallucinate success. Look closely for normal physics, normal shapes, or background inconsistencies.

# You MUST output ONLY a valid JSON object. 
# CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
# Example format:
# {
# "1": {
#     "reasoning": "I observe that while the main object is altered, the shadow is normal. Also, there are standard spherical bubbles on the right side, which violates the flat requirement.",
#     "score": 0.0
# }
# }"""
#         user_content_text = f"""Please evaluate the image strictly against the following criteria. 
# For each question, actively look for visual evidence that the image FAILS the criteria. If there is any ambiguity, normal physics, or partial failure, score it harshly.

# Questions and Criteria:
# {base_q_text}"""

#     MAX_RETRIES = 10  # 修改为重试 10 次
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": user_content_text},
#                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
#                     ]}
#                 ],
#                 temperature=0.0,
#                 max_tokens=5500
#             )
#             content = response.choices[0].message.content

#             print(f"\n[Debug 原始输出] 尝试次数 {attempt+1}:")
#             print(content)
#             print("-" * 40)

#             parsed_result = extract_json_robust(content)
            
#             if not parsed_result:
#                 print(f"      ⚠️ 解析为空，重试 ({attempt+1}/{MAX_RETRIES})...")
#                 time.sleep(5) # 解析失败稍微等一下再试
#                 continue 
                
#             batch_results = []
#             for temp_id, original_q in temp_id_map.items():
#                 res = parsed_result.get(temp_id, {})
#                 if not res: res = parsed_result.get(f"Question ID {temp_id}", {})
                
#                 batch_results.append({
#                     "id": original_q.get("id"),
#                     "source_id": original_q.get("source_id"),
#                     "prompt_level": original_q.get("prompt_level"),
#                     "question": original_q.get("question"),
#                     "weight": original_q.get("weight", 1),
#                     "score": float(res.get("score", 0.0)),
#                     "reasoning": res.get("reasoning", "Parsed"),
#                     "scoring_model": MODEL_NAME
#                 })
#             return batch_results

#         # 🔥 修改点：捕获所有异常并强制重试，而不是直接 return None
#         except Exception as e:
#             wait_time = 15  # 发生错误后等待 15 秒再重试 (您可以根据需要调整)
#             print(f"      ⚠️ 发生错误 ({e.__class__.__name__}: {e})，等待 {wait_time}s 后重试... ({attempt+1}/{MAX_RETRIES})")
#             time.sleep(wait_time)

#     print("      ❌ 达到最大重试 10 次，彻底放弃此图。")
#     return None
# # ================= 🚀 主程序 =================

# def main():
#     client = init_client()
    
#     print(f"🔍 正在扫描评测文件: {EVAL_ROOT_DIR}/**/*.json")
#     json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
#     if not json_files:
#         print("❌ 未找到任何 JSON 评测文件。")
#         return

#     if not os.path.exists(OUTPUT_BASE_DIR):
#         print(f"❌ 基础输出目录不存在: {OUTPUT_BASE_DIR}")
#         return
        
#     models = [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]
#     print(f"📦 发现 {len(models)} 个模型待处理: {models}")

#     for model_name in models:
#         print("\n" + "★"*60)
#         print(f"🚀 开始处理模型: {model_name}")
#         print("★"*60)
        
#         current_image_root = os.path.join(OUTPUT_BASE_DIR, model_name)
#         current_output_root = os.path.join(SCORE_BASE_DIR, f"gemini-{model_name}")
        
#         all_files_statistics = []
#         all_images_global = []

#         for idx, json_file in enumerate(json_files):
#             print(f"\n📂 [{model_name}] 处理文件 [{idx+1}/{len(json_files)}]: {os.path.basename(json_file)}")
            
#             image_dir, rel_path = get_image_dir_from_json_path(json_file, current_image_root)
#             output_file = os.path.join(current_output_root, rel_path.replace(".json", "_scores.json"))
            
#             final_results = []
#             processed_keys = set() 
#             l1_score_map = {} 
            
#             # 1. 读取已有的结果 (断点续传)
#             if os.path.exists(output_file):
#                 try:
#                     with open(output_file, 'r', encoding='utf-8') as f:
#                         final_results = json.load(f)
                        
#                         temp_l1_results = {}
#                         for item in final_results:
#                             sid = str(item.get('source_id'))
#                             lvl = str(item.get('prompt_level')).lower()
#                             processed_keys.add((sid, lvl))
                            
#                             if lvl == 'l1':
#                                 if sid not in temp_l1_results: temp_l1_results[sid] = []
#                                 temp_l1_results[sid].append(item)
                        
#                         for sid, res_list in temp_l1_results.items():
#                             l1_score_map[sid] = calculate_image_score(res_list)
                            
#                     print(f"⏩ 已加载现有进度: {len(processed_keys)} 张图片已处理")
#                 except Exception as e:
#                     print(f"⚠️ 读取现有文件出错 ({e})，将重新开始。")
#                     final_results = []

#             if not os.path.exists(image_dir):
#                 print(f"⚠️  图片目录不存在: {image_dir} (跳过)")
#                 continue
            
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
#             with open(json_file, 'r') as f:
#                 raw_data = json.load(f)
            
#             # 2. 按 source_id 分组任务
#             tasks_by_source = {}
#             for item in raw_data:
#                 sid = str(item['source_id'])
#                 lvl = str(item['prompt_level']).lower()
#                 if sid not in tasks_by_source:
#                     tasks_by_source[sid] = {}
#                 if lvl not in tasks_by_source[sid]:
#                     tasks_by_source[sid][lvl] = []
#                 tasks_by_source[sid][lvl].append(item)
            
#             # ================= 🔥 新增：自动克隆 L2 任务作为 L3 任务 =================
#             for sid, levels_dict in tasks_by_source.items():
#                 if 'l2' in levels_dict and 'l3' not in levels_dict:
#                     l3_tasks = []
#                     for q in levels_dict['l2']:
#                         q_clone = q.copy()
#                         q_clone['prompt_level'] = 'l3'  # 强制标记为 l3
#                         l3_tasks.append(q_clone)
#                     levels_dict['l3'] = l3_tasks
#             # =========================================================================

#             print(f"📊 总任务数: {len(tasks_by_source)} 个 Source ID")
            
#             # 3. 按 source_id 遍历，并强制按 l1 -> l2 -> l3 顺序处理
#             for i, (sid, levels_dict) in enumerate(tasks_by_source.items()):
#                 sorted_levels = sorted(levels_dict.keys(), key=lambda x: {'l1': 1, 'l2': 2, 'l3': 3}.get(x, 99))
                
#                 for level in sorted_levels:
#                     questions = levels_dict[level]
#                     img_name = f"{sid}_{level}.png"
                    
#                     if (sid, level) in processed_keys:
#                         continue

#                     img_path = os.path.join(image_dir, img_name)
                    
#                     # ================= 🔥 新增：图片格式兼容兜底 =================
#                     if not os.path.exists(img_path):
#                         if os.path.exists(img_path.replace(".png", ".jpg")):
#                             img_path = img_path.replace(".png", ".jpg")
#                     # =============================================================

#                     print(f"   [{i+1}/{len(tasks_by_source)}] 处理: {os.path.basename(img_path)} ...", end="", flush=True)
                    
#                     batch_res = []
                    
#                     # --- 核心逻辑：检查是否需要跳过 L2/L3 ---
#                     if level in ['l2', 'l3']:
#                         l1_score = l1_score_map.get(sid, 0.0)
#                         if l1_score < 0.5:
#                             print(f" ⏭️ 跳过 (L1分数为 {l1_score:.2f} < 0.5，直接赋0分)")
#                             for q in questions:
#                                 res = q.copy()
#                                 res.update({
#                                     "score": 0.0, 
#                                     "reasoning": "Skipped due to L1 score < 0.5", 
#                                     "scoring_model": "None"
#                                 })
#                                 batch_res.append(res)
                    
#                     # 如果没有被跳过，则正常处理
#                     if not batch_res:
#                         if not os.path.exists(img_path):
#                             print(" ⚠️ 缺失 (0分)")
#                             for q in questions:
#                                 res = q.copy()
#                                 res.update({"score": 0.0, "reasoning": "Image missing", "scoring_model": "None"})
#                                 batch_res.append(res)
#                         else:
#                             start_t = time.time()
#                             api_res = process_single_image_batch(client, img_path, questions, level)
#                             dur = time.time() - start_t
                            
#                             if api_res:
#                                 print(f" ✅ 完成 ({dur:.2f}s)")
#                                 batch_res.extend(api_res)
#                             else:
#                                 print(f" ❌ 失败 (0分)")
#                                 for q in questions:
#                                     res = q.copy()
#                                     res.update({"score": 0.0, "reasoning": "API Failed", "scoring_model": MODEL_NAME})
#                                     batch_res.append(res)
                    
#                     # --- 核心逻辑：计算并注入 image_score 和 l1_score ---
#                     current_image_score = calculate_image_score(batch_res)
                    
#                     if level == 'l1':
#                         l1_score_map[sid] = current_image_score
                    
#                     current_l1_score = l1_score_map.get(sid, 0.0)
                    
#                     for item in batch_res:
#                         item['image_score'] = round(current_image_score, 4)
#                         item['l1_score'] = round(current_l1_score, 4)
                        
#                     final_results.extend(batch_res)
#                     processed_keys.add((sid, level))
                    
#                     try:
#                         with open(output_file, 'w', encoding='utf-8') as f:
#                             json.dump(final_results, f, indent=2, ensure_ascii=False)
#                     except Exception as e:
#                         print(f"   ⚠️ 保存失败: {e}")

#                     time.sleep(0.5)

#             if final_results:
#                 file_images = process_image_scores(final_results)
#                 file_stats = calculate_stats_from_images(file_images, os.path.basename(json_file))
#                 all_files_statistics.append(file_stats)
#                 all_images_global.extend(file_images)
                
#                 print(f"📈 [统计] Overall: {file_stats['overall']} (L1:{file_stats.get('l1')} L2:{file_stats.get('l2')} L3:{file_stats.get('l3')})")

#         print(f"\n🌍 计算 [{model_name}] 全局统计 (Global Statistics)...")
#         if all_images_global:
#             global_stats = calculate_stats_from_images(all_images_global, "TOTAL_AVERAGE")
#             all_files_statistics.append(global_stats)
#             print(f"   🔹 TOTAL L1: {global_stats['l1']}")
#             print(f"   🔹 TOTAL L2: {global_stats['l2']}")
#             print(f"   🔹 TOTAL L3: {global_stats['l3']}")
#             print(f"   🔹 TOTAL Overall: {global_stats['overall']}")
#         else:
#             print("⚠️ 没有产生任何有效数据。")

#         summary_path = os.path.join(current_output_root, "summary_stats.json")
#         with open(summary_path, 'w', encoding='utf-8') as f:
#             json.dump(all_files_statistics, f, indent=2, ensure_ascii=False)
        
#         print(f"🎉 模型 {model_name} 处理完成！汇总已保存至: {summary_path}\n")

#     print("🏁 所有模型均已处理完毕！")

# if __name__ == "__main__":
#     main()
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
import argparse  # 新增：用于接收命令行参数

try:
    from PIL import Image
except ImportError:
    Image = None

# ================= ⚙️ 配置区域 =================

# 1. API 配置
#实验室的
# BASE_URL = "http://35.220.164.252:3888/v1"
# API_KEY = "sk-aEgLZS952k0LiNx7kJYJOo5tkhOERlNQfxhnV5xIugxAzltm"

#神马
# BASE_URL = "https://api.whatai.cc/v1"
# API_KEY = "sk-o6mBpaewBQ1TvcoQETr2mwpl6kcLIG4j59QQymhnwdNAUOtw"

#fchat ai
BASE_URL = "http://fchatapi.dykyzdh.top/v1"
API_KEY = "sk-FNAqlNwvS9jwtBF5gCGK1WOqeXJ9ALeIDbULLAAVg8Lwsbfm"

MODEL_NAME = "gemini-3-pro-preview"
print(BASE_URL)
# 2. 路径配置
EVAL_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_v5_fixed"
OUTPUT_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output"
SCORE_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score"

# ================= 🛠️ 工具函数 =================

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
        print(f"❌ 读取图片失败 {image_path}: {e}")
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

# --- 统计逻辑 ---

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

# ================= 🧠 核心逻辑 =================

# def process_single_image_batch(client, image_path, questions_list, level):
#     base64_image = encode_image(image_path)
#     if not base64_image: return None

#     base_q_text = ""
#     temp_id_map = {}
#     for idx, q in enumerate(questions_list):
#         temp_id = str(idx + 1)
#         temp_id_map[temp_id] = q
#         base_q_text += f'\nQuestion ID "{temp_id}":\nQuestion: {q["question"]}\nCriteria: {q["evaluation_criteria"]}\n---'

#     if level.lower() == 'l1':
#         system_prompt = """You are an objective and balanced Image Quality Assurance Assistant. Your job is to evaluate whether an AI-generated image accurately reflects standard real-world physics, common sense, and the provided factual criteria.
# Note: Minor AI artifacts or slight imperfections are acceptable as long as the main subject and overall scene align with the prompt and basic reality. Do not be overly harsh on negligible details.

# You MUST output ONLY a valid JSON object. 
# CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
# Example format:
# {
# "1": {
#     "reasoning": "The image clearly shows a normal apple on a table. The lighting and shadows are realistic. There are some minor blurry pixels in the background, but it does not affect the overall factual accuracy.",
#     "score": 1.0
# }
# }"""
#         user_content_text = f"""Please evaluate the image based on the following criteria. 
# Assess whether the image generally satisfies the factual requirements and common sense. If the criteria are mostly met despite minor AI flaws, you can give a high score (e.g., 1.0). If it partially fails or has noticeable logical errors, score it accordingly (e.g., 0.5). Only give 0.0 if it completely fails the criteria.

# Questions and Criteria:
# {base_q_text}"""
#     else:
#         system_prompt = """You are a strict, adversarial Image Quality Assurance Judge. Your primary job is to FIND FLAWS and penalize AI-generated images that fail to strictly follow counterfactual physics or logic.
# WARNING: AI models often generate normal objects instead of the requested counterfactual ones. Do NOT hallucinate success. Look closely for normal physics, normal shapes, or background inconsistencies.

# You MUST output ONLY a valid JSON object. 
# CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
# Example format:
# {
# "1": {
#     "reasoning": "I observe that while the main object is altered, the shadow is normal. Also, there are standard spherical bubbles on the right side, which violates the flat requirement.",
#     "score": 0.0
# }
# }"""
#         user_content_text = f"""Please evaluate the image strictly against the following criteria. 
# For each question, actively look for visual evidence that the image FAILS the criteria. If there is any ambiguity, normal physics, or partial failure, score it harshly.

# Questions and Criteria:
# {base_q_text}"""

#     MAX_RETRIES = 10
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": [
#                         {"type": "text", "text": user_content_text},
#                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
#                     ]}
#                 ],
#                 temperature=0.0,
#                 max_tokens=5500
#             )
#             content = response.choices[0].message.content

#             print(f"\n[Debug 原始输出] 尝试次数 {attempt+1}:")
#             print(content)
#             print("-" * 40)

#             parsed_result = extract_json_robust(content)
            
#             if not parsed_result:
#                 print(f"      ⚠️ 解析为空，重试 ({attempt+1}/{MAX_RETRIES})...")
#                 time.sleep(5)
#                 continue 
                
#             batch_results = []
#             for temp_id, original_q in temp_id_map.items():
#                 res = parsed_result.get(temp_id, {})
#                 if not res: res = parsed_result.get(f"Question ID {temp_id}", {})
                
#                 batch_results.append({
#                     "id": original_q.get("id"),
#                     "source_id": original_q.get("source_id"),
#                     "prompt_level": original_q.get("prompt_level"),
#                     "question": original_q.get("question"),
#                     "weight": original_q.get("weight", 1),
#                     "score": float(res.get("score", 0.0)),
#                     "reasoning": res.get("reasoning", "Parsed"),
#                     "scoring_model": MODEL_NAME
#                 })
#             return batch_results

#         except Exception as e:
#             wait_time = 15
#             print(f"      ⚠️ 发生错误 ({e.__class__.__name__}: {e})，等待 {wait_time}s 后重试... ({attempt+1}/{MAX_RETRIES})")
#             time.sleep(wait_time)

#     print("      ❌ 达到最大重试 10 次，彻底放弃此图。")
#     return None

##宽容版本
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
        # --- L1: 极度宽容的基础事实与常识评估 ---
        system_prompt = """You are a highly forgiving Image Quality Assurance Assistant. Your job is to evaluate whether an AI-generated image generally captures the main idea of the provided factual criteria and common sense.
Note: AI images naturally have artifacts, weird textures, or minor background inconsistencies. You should completely ignore these standard AI flaws as long as the main subject is recognizable and aligns with the prompt. Default to giving a high score (0.8 - 1.0) if the primary subject and scene are mostly correct.

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
Be very lenient. If the image successfully conveys the core concept requested, give it a 1.0. Deduct points (e.g., 0.5 to 0.8) only if there are significant missing elements or major deviations from the prompt. Give 0.0 only if the image is completely unrelated to the criteria.

Questions and Criteria:
{base_q_text}"""
    else:
        # --- L2/L3: 理性的反事实评估与弹性打分 ---
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

            print(f"\n[Debug 原始输出] 尝试次数 {attempt+1}:")
            print(content)
            print("-" * 40)

            parsed_result = extract_json_robust(content)
            
            if not parsed_result:
                print(f"      ⚠️ 解析为空，重试 ({attempt+1}/{MAX_RETRIES})...")
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
            print(f"      ⚠️ 发生错误 ({e.__class__.__name__}: {e})，等待 {wait_time}s 后重试... ({attempt+1}/{MAX_RETRIES})")
            time.sleep(wait_time)

    print("      ❌ 达到最大重试 10 次，彻底放弃此图。")
    return None


# ================= 🚀 主程序（支持指定 model） =================

def main():
    # ================= 新增：解析命令行参数 =================
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
    
    print(f"🔍 正在扫描评测文件: {EVAL_ROOT_DIR}/**/*.json")
    json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
    if not json_files:
        print("❌ 未找到任何 JSON 评测文件。")
        return

    if not os.path.exists(OUTPUT_BASE_DIR):
        print(f"❌ 基础输出目录不存在: {OUTPUT_BASE_DIR}")
        return
        
    # ================= 新增：支持指定 model =================
    all_models = [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]
    
    if args.model:
        # 用户指定了 model，只处理这一个
        if args.model not in all_models:
            print(f"❌ 指定的模型不存在！可用模型：{all_models}")
            return
        models = [args.model]
        print(f"🚀 仅处理指定模型: {args.model}")
    else:
        # 不指定，处理全部
        models = all_models
        print(f"📦 发现 {len(models)} 个模型待处理: {models}")
    # ========================================================

    for model_name in models:
        print("\n" + "★"*60)
        print(f"🚀 开始处理模型: {model_name}")
        print("★"*60)
        
        current_image_root = os.path.join(OUTPUT_BASE_DIR, model_name)
        current_output_root = os.path.join(SCORE_BASE_DIR, f"gemini-{model_name}")
        
        all_files_statistics = []
        all_images_global = []

        for idx, json_file in enumerate(json_files):
            print(f"\n📂 [{model_name}] 处理文件 [{idx+1}/{len(json_files)}]: {os.path.basename(json_file)}")
            
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
                            
                    print(f"⏩ 已加载现有进度: {len(processed_keys)} 张图片已处理")
                except Exception as e:
                    print(f"⚠️ 读取现有文件出错 ({e})，将重新开始。")
                    final_results = []

            if not os.path.exists(image_dir):
                print(f"⚠️  图片目录不存在: {image_dir} (跳过)")
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
            
            # 自动克隆 L2 → L3
            for sid, levels_dict in tasks_by_source.items():
                if 'l2' in levels_dict and 'l3' not in levels_dict:
                    l3_tasks = []
                    for q in levels_dict['l2']:
                        q_clone = q.copy()
                        q_clone['prompt_level'] = 'l3'
                        l3_tasks.append(q_clone)
                    levels_dict['l3'] = l3_tasks

            print(f"📊 总任务数: {len(tasks_by_source)} 个 Source ID")
            
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

                    print(f"   [{i+1}/{len(tasks_by_source)}] 处理: {os.path.basename(img_path)} ...", end="", flush=True)
                    
                    batch_res = []
                    
                    if level in ['l2', 'l3']:
                        l1_score = l1_score_map.get(sid, 0.0)
                        if l1_score < 0.5:
                            print(f" ⏭️ 跳过 (L1分数为 {l1_score:.2f} < 0.5，直接赋0分)")
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
                            print(" ⚠️ 缺失 (0分)")
                            for q in questions:
                                res = q.copy()
                                res.update({"score": 0.0, "reasoning": "Image missing", "scoring_model": "None"})
                                batch_res.append(res)
                        else:
                            start_t = time.time()
                            api_res = process_single_image_batch(client, img_path, questions, level)
                            dur = time.time() - start_t
                            
                            if api_res:
                                print(f" ✅ 完成 ({dur:.2f}s)")
                                batch_res.extend(api_res)
                            else:
                                print(f" ❌ 失败 (0分)")
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
                        print(f"   ⚠️ 保存失败: {e}")

                    time.sleep(0.5)

            if final_results:
                file_images = process_image_scores(final_results)
                file_stats = calculate_stats_from_images(file_images, os.path.basename(json_file))
                all_files_statistics.append(file_stats)
                all_images_global.extend(file_images)
                
                print(f"📈 [统计] Overall: {file_stats['overall']} (L1:{file_stats.get('l1')} L2:{file_stats.get('l2')} L3:{file_stats.get('l3')})")

        print(f"\n🌍 计算 [{model_name}] 全局统计 (Global Statistics)...")
        if all_images_global:
            global_stats = calculate_stats_from_images(all_images_global, "TOTAL_AVERAGE")
            all_files_statistics.append(global_stats)
            print(f"   🔹 TOTAL L1: {global_stats['l1']}")
            print(f"   🔹 TOTAL L2: {global_stats['l2']}")
            print(f"   🔹 TOTAL L3: {global_stats['l3']}")
            print(f"   🔹 TOTAL Overall: {global_stats['overall']}")
        else:
            print("⚠️ 没有产生任何有效数据。")

        summary_path = os.path.join(current_output_root, "summary_stats.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_files_statistics, f, indent=2, ensure_ascii=False)
        
        print(f"🎉 模型 {model_name} 处理完成！汇总已保存至: {summary_path}\n")

    print("🏁 所有模型均已处理完毕！")

if __name__ == "__main__":
    main()