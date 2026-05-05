#!/bin/bash

# 1. 初始化 Conda 环境配置
source /home/leijiayi/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate /mnt/shared-storage-user/leijiayi/envs/qwen

# 3. 执行 Python 脚本
python scripts/score/qwen3vl-235b_single.py
