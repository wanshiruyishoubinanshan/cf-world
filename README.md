# 🦃 CF-World: Are Text-to-Image Models Inductivist Turkeys?

[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026-blue.svg)](link_to_paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CF-World (Counterfactual-World)** is a novel benchmark designed to probe whether text-to-image (T2I) models possess genuine causal understanding or if they are merely sophisticated pattern matchers—much like Bertrand Russell's "Inductivist Turkey." 

Current T2I models generate stunning factual images, but do they understand the underlying physical laws? CF-World systematically evaluates this by forcing models to generate images under rules that contradict real-world priors.

### 🌟 核心亮点 (Key Highlights)
- **三级递进评测 (Three-Level Progressive Framework):** 
  - **L1 (Factual):** 基础事实生成。
  - **L2 (Explicit Counterfactual):** 显式反事实生成（提供反事实条件和视觉结果）。
  - **L3 (Implicit Counterfactual):** 隐式反事实生成（仅提供条件，要求模型自主推理）。
- **创新指标 (Novel Metrics):** 引入 **PRR** (Prior Resistance Rate) 和 **RRR** (Reasoning Retention Rate) 来量化模型克服先验偏见和保持推理的能力。
- **深度归因分析 (Deep Attribution Analysis):** 包含因果解耦 (Causal Decoupling)、属性解耦 (Attribute Decoupling) 和去名词化 (De-nominalization/De-norm) 实验。

---

## 📂 目录结构 (Repository Structure)

代码库被清晰地划分为提示词、评测问题和执行脚本三个主要部分：

```text
├── eval_questions/        # 预生成的评测问题 (基于不同学科，如物理、生物等)
│   ├── physics/           # 物理学细分领域 (Astronomy, Mechanics, etc.)
│   └── ...
├── prompt/                # 原始的基础 Prompt 数据
│   ├── physics/
│   └── ...
└── scripts/               # 核心执行脚本
    ├── generate_eval/     # 生成评测问题的脚本
    │   ├── gemini.py      # 使用 Gemini 生成标准评测问题
    │   └── rule_decouple.py # 生成规则解耦 (Causal Decoupling) 的评测问题
    └── score/             # 自动化打分脚本
        ├── gemini.py      # 使用 Gemini 进行标准打分
        ├── qwen3vl-235b.py# 使用 Qwen3-VL 进行标准打分
        ├── rule_decouple.py # 规则解耦实验打分
        ├── attribute_decouple.py # 属性解耦实验打分
        └── denorm.py      # 去名词化 (De-norm) 实验打分
```

---

## 🚀 快速开始 (Getting Started)

### 1. 环境准备
请确保安装了必要的依赖项（如 `vllm`, `transformers`, `Pillow` 等）：
```bash
pip install -r requirements.txt
```

### 2. 生成评测问题 (Generate Evaluation Questions)
如果你想基于 `prompt/` 目录下的基础数据重新生成评测问题，请运行：
```bash
# 生成标准 CF-World 评测问题
python scripts/generate_eval/gemini.py

# 生成规则解耦 (Rule Decouple) 专属评测问题
python scripts/generate_eval/rule_decouple.py
```

### 3. 自动化打分 (Automated Scoring)
在模型生成图片后，你可以使用强大的 VLM（如 Qwen3-VL 或 Gemini）进行多维度自动化打分：

```bash
# 使用 Qwen3-VL-235B 进行标准三级 (L1/L2/L3) 打分
python scripts/score/qwen3vl-235b.py

# 使用 Gemini 进行标准打分
python scripts/score/gemini.py
```

### 4. 深度分析实验打分 (Deep Analysis Scoring)
针对论文中的三大深度分析实验，我们提供了专门的打分脚本：
```bash
# 1. 规则/因果解耦 (Causal Decoupling)
python scripts/score/rule_decouple.py

# 2. 属性解耦 (Attribute Decoupling)
python scripts/score/attribute_decouple.py

# 3. 去名词化 (De-nominalization / De-norm)
python scripts/score/denorm.py
```

---

## 📊 评测指标说明 (Metrics Explained)

| **Metric** | **Formula** | **Description** |
|------------|-------------|-----------------|
| **PRR** (Prior Resistance Rate) | $$S_{L2} / S_{L1}$$ | 衡量模型克服现实世界常识先验，遵循显式反事实指令的能力。 |
| **RRR** (Reasoning Retention Rate) | $$(S_{L3} / S_{L2}) \times S_{L3}$$ | 量化模型在缺乏显式视觉提示时，自主进行因果推理并保持生成质量的鲁棒性。 |

> **注意:** 评分系统包含严格的阈值机制（Sequential Thresholding）。如果 L1 基础事实得分 $$S_{L1} < 0.5$$，则对应的 L2 和 L3 得分将严格记为 0。

---

## 💡 引用 (Citation)
如果你在研究中使用了本数据集或代码，请引用我们的论文：

```bibtex
@article{cfworld2026,
  title={Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning},
  author={Anonymous Author(s)},
  journal={NeurIPS},
  year={2026}
}
```

---

你可以直接将这段 Markdown 复制到你的 `README.md` 文件中。如果有任何具体的模块说明需要微调（比如增加环境配置的细节，或者修改具体的命令行参数），随时告诉我！
