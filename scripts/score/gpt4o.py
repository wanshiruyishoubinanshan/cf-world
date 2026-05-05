# import os
import json
import glob
import base64
import time
import httpx
from pathlib import Path
from openai import OpenAI
import os

# ================= ⚙️ 配置区域 =================

# 1. 路径配置
# 评测问题根目录 (输入 1)
EVAL_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/eval"
# 图片输入根目录 (输入 2)
IMAGE_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/output/janus-pro-"
# 结果输出根目录 (输出)
OUTPUT_ROOT_DIR = "/mnt/shared-storage-user/leijiayi/counterfactual/score/gpt4o"

# 2. API 配置 (沿用参考代码)
BASE_URL = "http://35.220.164.252:3888/v1"
API_KEY = "sk-aEgLZS952k0LiNx7kJYJOo5tkhOERlNQfxhnV5xIugxAzltm"

# 配置 HTTP 客户端 (关闭 SSL 验证，增加超时时间)
http_client = httpx.Client(timeout=120.0, verify=False)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# ================= 🛠️ 工具函数 =================

def encode_image(image_path):
    """将图片转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_paths_from_json(json_path):
    """
    根据 JSON 路径推导图片目录和输出路径
    规则: eval/gemini/physics/Astronomy_gemini.json 
    -> 图片: output/sd3.5/physics/Astronomy
    -> 输出: score/gpt4o/physics/Astronomy_gpt4o.json
    """
    # 1. 计算相对路径 (e.g., "physics/Astronomy_gemini.json")
    rel_path = os.path.relpath(json_path, EVAL_ROOT_DIR)
    path_obj = Path(rel_path)
    
    # 2. 解析目录和文件名
    parent_dir = path_obj.parent  # "physics"
    stem = path_obj.stem.replace("_gemini", "")  # "Astronomy"
    
    # 3. 构建目标图片目录
    target_image_dir = os.path.join(IMAGE_ROOT_DIR, parent_dir, stem)
    
    # 4. 构建输出文件路径
    output_filename = rel_path.replace("_gemini.json", "_gpt4o.json")
    target_output_file = os.path.join(OUTPUT_ROOT_DIR, output_filename)
    
    return target_image_dir, target_output_file

def calculate_statistics(result_filepath):
    """计算单个文件的简单统计"""
    if not os.path.exists(result_filepath): return
    try:
        with open(result_filepath, 'r') as f: data = json.load(f)
        scores = [item.get('score', 0.0) for item in data]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"📊 [统计] 文件平均分: {avg:.4f} (样本数: {len(scores)})")
    except:
        pass

# ================= 🧠 GPT 调用逻辑 =================

def call_gpt4o_vision(image_base64, questions_list, max_retries=5):
    """调用 GPT-4o Vision，带有重试机制"""
    
    # 1. 拼接 Prompt
    questions_text = ""
    for idx, q in enumerate(questions_list):
        # 使用临时 ID (1, 2, 3...) 简化 Prompt，减少 Token 消耗
        questions_text += f'\nQuestion ID "{idx+1}":\nQuestion: {q["question"]}\nCriteria: {q["evaluation_criteria"]}\n---'

    system_prompt = """You are an expert Image Quality Assurance Specialist. 
Evaluate the image based strictly on the provided questions.
For each Question ID, provide:
1. "score": 1.0 (Pass), 0.5 (Partial), 0.0 (Fail).
2. "reasoning": Brief explanation.
Output ONLY a JSON object. Example:
{"1": {"score": 1.0, "reasoning": "Visible."}}"""

    # 2. 重试循环
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Evaluate criteria:\n{questions_text}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"      ⚠️ API Error (Attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1)) # 指数退避
            else:
                print("      ❌ Max retries reached.")
                return None
    return None

# ================= 🚀 主程序 =================

def main():
    # 1. 扫描所有 JSON 文件
    print(f"🔍 正在扫描评测文件: {EVAL_ROOT_DIR}/**/*.json")
    json_files = glob.glob(os.path.join(EVAL_ROOT_DIR, "**/*.json"), recursive=True)
    
    if not json_files:
        print("❌ 未找到任何 JSON 文件，请检查路径。")
        return
    
    print(f"📋 找到 {len(json_files)} 个评测文件，准备开始处理...")

    # 2. 遍历处理每个文件
    for json_idx, json_file in enumerate(json_files):
        print(f"\n" + "="*60)
        print(f"📂 处理文件 [{json_idx+1}/{len(json_files)}]: {os.path.basename(json_file)}")
        
        # 2.1 路径解析与检查
        image_dir, output_file = get_paths_from_json(json_file)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 断点续打：如果结果已存在，跳过
        if os.path.exists(output_file):
            print(f"⏩ 结果已存在，跳过: {output_file}")
            continue

        print(f"🖼️  对应图片目录: {image_dir}")
        if not os.path.exists(image_dir):
            print(f"⚠️  警告: 图片目录不存在，跳过此文件！")
            continue

        # 2.2 加载评测数据并按图片分组
        with open(json_file, 'r') as f:
            eval_data = json.load(f)

        image_tasks = {}
        for item in eval_data:
            key = (item['source_id'], item['prompt_level'])
            if key not in image_tasks: image_tasks[key] = []
            image_tasks[key].append(item)

        final_results_list = []
        
        # 2.3 逐张图片处理
        total_imgs = len(image_tasks)
        for i, ((source_id, level), questions) in enumerate(image_tasks.items()):
            image_filename = f"{source_id}_{level}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            # 进度条效果
            print(f"   [{i+1}/{total_imgs}] 正在评估: {image_filename} ...", end="\r")

            # 检查图片是否存在
            if not os.path.exists(image_path):
                # 尝试 jpg
                if os.path.exists(image_path.replace(".png", ".jpg")):
                    image_path = image_path.replace(".png", ".jpg")
                else:
                    print(f"\n      ❌ 图片缺失: {image_filename}")
                    continue

            # 调用 GPT-4o
            gpt_response = None
            try:
                b64_img = encode_image(image_path)
                gpt_response = call_gpt4o_vision(b64_img, questions)
            except Exception as e:
                print(f"\n      ❌ 处理异常: {e}")

            # 结果解析与存储
            for idx, q in enumerate(questions):
                temp_id = str(idx + 1) # 对应 Prompt 中的 ID
                
                # 默认值 (如果 GPT 失败或没返回该 ID)
                score = 0.0
                reasoning = "Error: API Failed or Missing Response"

                if gpt_response:
                    res = gpt_response.get(temp_id, {})
                    if not res: res = gpt_response.get(f"Question ID {temp_id}", {})
                    score = float(res.get("score", 0.0))
                    reasoning = res.get("reasoning", reasoning)

                final_results_list.append({
                    "id": q.get("id"),
                    "source_id": source_id,
                    "prompt_level": level,
                    "image_filename": image_filename,
                    "question": q['question'],
                    "weight": q.get('weight', 1),
                    "score": score,
                    "reasoning": reasoning,
                    "evaluator": "GPT-4o"
                })
            
            # 避免 API 速率限制，轻微延时
            time.sleep(0.5)

        # 2.4 保存当前 JSON 文件的结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results_list, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 已保存: {output_file}")
        calculate_statistics(output_file)

    print("\n" + "="*60)
    print("🎉 所有文件评估完成！")

if __name__ == "__main__":
    main()
