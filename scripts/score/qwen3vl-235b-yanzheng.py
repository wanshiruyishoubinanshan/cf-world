import os
import json
import re
import traceback
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ================= ⚙️ 配置区域 =================

# 1. 模型路径 (Qwen3-VL-235B)
MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct-FP8/snapshots/d464a056915e088a7621533813ed553ceea73a6e"

# 2. 路径配置
EVAL_JSON_PATH = "/mnt/shared-storage-user/leijiayi/counterfactual/eval_yanzheng/rule_decouple_eval_questions.json"
IMAGE_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output/yanzheng"  # 存放各个模型文件夹的根目录
OUTPUT_BASE_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score_yanzheng_rule"

# 3. 硬件与推理配置
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 8192
BATCH_SIZE_PER_SAVE = 32

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

def calculate_scores(results_list):
    """
    计算分数：
    1. 每张图 (source_id) 的得分为内部 questions 的加权平均。
    2. 根据分类 (original / cf) 分别统计平均分。
    3. 计算模型整体的总平均分。
    """
    grouped_data = {}
    for item in results_list:
        sid = str(item.get('source_id'))
        if sid not in grouped_data:
            grouped_data[sid] = []
        grouped_data[sid].append(item)
    
    image_scores = []
    category_scores = {'original': [], 'cf': []}
    
    for sid, items in grouped_data.items():
        # 计算单张图片的加权平均分
        w_sum = sum(float(x.get('score', 0)) * float(x.get('weight', 1)) for x in items)
        total_w = sum(float(x.get('weight', 1)) for x in items)
        img_score = w_sum / total_w if total_w > 0 else 0.0
        
        # 提取该图片的类别 (同一 source_id 下的 category 是一致的)
        cat = str(items[0].get('category', '')).lower()
        
        # 将图片得分写回每个 question 记录中
        for x in items:
            x['image_score'] = round(img_score, 4)
            
        image_scores.append(img_score)
        if cat in category_scores:
            category_scores[cat].append(img_score)
        
    # 计算各项平均分
    overall_score = sum(image_scores) / len(image_scores) if image_scores else 0.0
    original_score = sum(category_scores['original']) / len(category_scores['original']) if category_scores['original'] else 0.0
    cf_score = sum(category_scores['cf']) / len(category_scores['cf']) if category_scores['cf'] else 0.0
    
    return results_list, round(overall_score, 4), round(original_score, 4), round(cf_score, 4)

# ================= 🚀 主逻辑 =================

def main():
    print(f"🔍 正在读取评测问题文件: {EVAL_JSON_PATH}")
    if not os.path.exists(EVAL_JSON_PATH):
        print("❌ 未找到 JSON 文件！")
        return
        
    with open(EVAL_JSON_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
        
    # 按 source_id 对问题进行分组
    questions_by_source_id = {}
    for item in eval_data:
        sid = str(item['source_id'])
        if sid not in questions_by_source_id:
            questions_by_source_id[sid] = []
        questions_by_source_id[sid].append(item)
        
    print(f"✅ 共加载了 {len(questions_by_source_id)} 个独立图片的评测任务。")

    if not os.path.exists(IMAGE_BASE_DIR):
        print(f"❌ 图片基础目录不存在: {IMAGE_BASE_DIR}")
        return
        
    # 获取所有需要评测的模型列表
    model_names = [d for d in os.listdir(IMAGE_BASE_DIR) if os.path.isdir(os.path.join(IMAGE_BASE_DIR, d))]
    model_names.sort()
    print(f"📦 发现 {len(model_names)} 个模型待评测: {', '.join(model_names)}")
    
    # 全局加载模型
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

    # ================= 🔄 外层循环：遍历所有模型 =================
    for model_idx, current_model in enumerate(model_names):
        print("\n" + "★"*80)
        print(f"🚀 开始处理模型 [{model_idx+1}/{len(model_names)}]: {current_model}")
        print("★"*80)
        
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, current_model)
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, "eval_results.json")

        final_results_list = []
        processed_sids = set()

        # A. 加载断点进度
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    final_results_list = json.load(f)
                for item in final_results_list:
                    processed_sids.add(str(item.get('source_id')))
                print(f"⏩ 已加载现有进度: {len(processed_sids)} 张图片")
            except Exception:
                final_results_list = []

        # B. 准备当前模型的任务
        tasks_to_run = []
        for sid, questions in questions_by_source_id.items():
            if sid in processed_sids: 
                continue
            
            img_name = f"{sid}.png"
            img_path = os.path.join(IMAGE_BASE_DIR, current_model, "rule_decouple", img_name)
            
            # 兼容 jpg
            if not os.path.exists(img_path):
                img_path_jpg = img_path.replace(".png", ".jpg")
                if os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                else:
                    # 图片缺失，直接记 0 分
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
                    "filename": os.path.basename(img_path),
                    "image_obj": image_obj,
                    "temp_id_map": temp_id_map,
                    "base_q_text": q_text,
                    "retry_count": 0
                })
            except Exception as e:
                print(f"   ❌ 图片读取错误 {img_path}: {e}")

        # C. 推理逻辑
        chunk_size = BATCH_SIZE_PER_SAVE
        total_chunks = (len(tasks_to_run) + chunk_size - 1) // chunk_size

        # 统一的 System Prompt
        unified_system_prompt = """You are an objective and strict Image Quality Assurance Judge. Your job is to evaluate whether an AI-generated image accurately reflects the provided criteria.
        Read the criteria carefully. If the image fails to meet the specific constraints, score it harshly (0.5 or 0.0) according to the strict criteria. If it perfectly meets them, score it highly (1.0).

        You MUST output ONLY a valid JSON object. 
        CRITICAL: You must generate the "reasoning" BEFORE the "score" to ensure you think before judging.
        Example format:
        {
          "1": {
            "reasoning": "The red arrow is clearly visible and points exactly to the bottom-right corner as requested.",
            "score": 1.0
          }
        }"""

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
                    user_content = f"""Please evaluate the image based on the following criteria. 
                    Assess whether the image satisfies the requirements. Be objective but strict.

                    Questions and Criteria:
                    {task['base_q_text']}"""

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
                                "rule_id": original_q.get("rule_id"),
                                "category": original_q.get("category"),
                                "question_type": original_q.get("question_type"),
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
                                    "score": 0.0, 
                                    "weight": original_q.get('weight', 1), 
                                    "reasoning": "FAILED PARSING"
                                })
                pending_in_batch = next_round_pending
            
            # 存盘
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)

        # D. 计算当前模型的最终得分
        if final_results_list:
            final_results_list, overall_score, original_score, cf_score = calculate_scores(final_results_list)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, ensure_ascii=False)
                
            model_summary = {
                "model_name": current_model,
                "overall_score": overall_score,
                "original_score": original_score,
                "cf_score": cf_score
            }
            global_comparison_stats.append(model_summary)
            print(f"📈 模型 [{current_model}] 成绩 -> Overall: {overall_score:.4f} | Original: {original_score:.4f} | CF: {cf_score:.4f}")
        else:
            print(f"⚠️ 模型 {current_model} 没有产生任何有效数据。")

    # ================= 🏆 最终步骤：生成全局对比汇总 =================
    print("\n" + "="*80)
    print("🏆 所有模型处理完毕！正在生成全局对比汇总表...")
    
    global_summary_path = os.path.join(OUTPUT_BASE_DIR, "global_models_comparison.json")
    global_comparison_stats.sort(key=lambda x: x.get('overall_score', 0.0), reverse=True)
    
    with open(global_summary_path, 'w', encoding='utf-8') as f:
        json.dump(global_comparison_stats, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 全局汇总已保存至: {global_summary_path}")
    print("排行榜预览:")
    for rank, stat in enumerate(global_comparison_stats):
        print(f"  {rank+1}. {stat['model_name']:<25} | Overall: {stat['overall_score']:.4f} | Original: {stat['original_score']:.4f} | CF: {stat['cf_score']:.4f}")

if __name__ == "__main__":
    main()
