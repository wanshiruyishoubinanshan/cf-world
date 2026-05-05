# 🦃 CF-World: Are Text-to-Image Models Inductivist Turkeys?

[![Paper](https://img.shields.io/badge/Paper-Under_Review-blue.svg)](link_to_paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CF-World (Counterfactual-World)** is a novel benchmark designed to probe whether text-to-image (T2I) models possess genuine causal understanding or if they are merely sophisticated pattern matchers—much like Bertrand Russell's "Inductivist Turkey." 

While current T2I models can generate stunning images that comply with everyday factual knowledge, they often fail catastrophically when physical laws are systematically altered. CF-World evaluates this by forcing models to generate images under rules that contradict real-world priors, revealing their true reasoning capabilities.

---

## 🌟 Key Highlights

- **Three-Level Progressive Framework:** 
  - **L1 (Factual):** Standard factual generation under ordinary world knowledge.
  - **L2 (Explicit Counterfactual):** Introduces a counterfactual condition and explicitly specifies the resulting visual state.
  - **L3 (Implicit Counterfactual):** Introduces only the counterfactual condition; the model must autonomously deduce the visual outcome.
- **Novel Diagnostic Metrics:** Introduces **PRR** (Prior Resistance Rate) and **RRR** (Reasoning Retention Rate) to quantify a model's ability to overcome entrenched priors and maintain logical deduction.
- **Deep Attribution Analysis:** Includes specialized evaluation tracks for Causal Decoupling, Attribute Decoupling, and De-nominalization (De-norm) to isolate the root causes of model failures.

---

## 📂 Repository Structure

The repository is organized into prompts, pre-generated evaluation questions, and execution scripts:

```text
├── eval_questions/        # Pre-generated evaluation questions (categorized by discipline)
│   ├── physics/           # Physics sub-disciplines (Astronomy, Mechanics, etc.)
│   └── ...
├── prompt/                # Raw base prompts and counterfactual rules
│   ├── physics/
│   └── ...
└── scripts/               # Core execution scripts
    ├── generate_eval/     # Scripts to generate evaluation questions
    │   ├── gemini.py      # Generates standard CF-World questions via Gemini
    │   └── rule_decouple.py # Generates questions for the Causal Decoupling experiment
    └── score/             # Automated VLM-based scoring scripts
        ├── gemini.py      # Standard multi-dimensional scoring using Gemini
        ├── qwen3vl-235b.py# Standard multi-dimensional scoring using Qwen3-VL
        ├── rule_decouple.py # Scoring for the Causal Decoupling experiment
        ├── attribute_decouple.py # Scoring for the Attribute Decoupling experiment
        └── denorm.py      # Scoring for the De-nominalization (De-norm) experiment
```

---

## 🚀 Getting Started

### 1. Environment Setup
Ensure you have the necessary dependencies installed (e.g., `vllm`, `transformers`, `Pillow`):
```bash
pip install -r requirements.txt
```

### 2. Generate Evaluation Questions
If you want to generate the evaluation questions from scratch based on the raw data in the `prompt/` directory, run the following:
```bash
# Generate standard CF-World evaluation questions
python scripts/generate_eval/gemini.py

# Generate specific evaluation questions for the Rule/Causal Decoupling experiment
python scripts/generate_eval/rule_decouple.py
```

### 3. Automated Scoring
Once your target T2I models have generated the images, use a Vision-Language Model (VLM) to evaluate them across our multi-dimensional criteria:

```bash
# Run standard 3-level (L1/L2/L3) evaluation using Qwen3-VL-235B
python scripts/score/qwen3vl-235b.py

# Run standard 3-level evaluation using Gemini
python scripts/score/gemini.py
```

### 4. Deep Analysis Scoring
To reproduce the deep attribution analysis experiments mentioned in the paper, use the dedicated scoring scripts:
```bash
# 1. Causal Decoupling Evaluation
python scripts/score/rule_decouple.py

# 2. Attribute Decoupling Evaluation
python scripts/score/attribute_decouple.py

# 3. De-nominalization (De-norm) Evaluation
python scripts/score/denorm.py
```

---

## 📊 Metrics Explained

| **Metric** | **Formula** | **Description** |
|------------|-------------|-----------------|
| **PRR** (Prior Resistance Rate) | $$S_{L2} / S_{L1}$$ | Measures the model's ability to resist real-world priors and follow explicit counterfactual instructions. |
| **RRR** (Reasoning Retention Rate) | $$(S_{L3} / S_{L2}) \times S_{L3}$$ | Quantifies the robustness of a model's causal reasoning when explicit visual cues are removed. |

> **Note:** The scoring system employs a **Sequential Thresholding Mechanism**. If a model fails the basic factual baseline ($$S_{L1} < 0.5$$), its corresponding L2 and L3 scores are strictly zeroed out to prevent false positives from random generation.
