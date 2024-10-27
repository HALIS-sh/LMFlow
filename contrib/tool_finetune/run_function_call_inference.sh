#!/bin/bash

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO

# 定义 num_gpus 和 gpu_memory_utilization 变量
NUM_GPUS=4
GPU_MEMORY_UTILIZATION=0.9

# 运行 Python 脚本，传递参数
# python ./contrib/tool-finetune/function_call_inference.py   --num-gpus $NUM_GPUS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
python ./contrib/tool-finetune/tool_inference.py