#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
model_name_or_path=gpt2
dataset_path=data/alpaca/train_conversation
output_dir=output_models/finetune
deepspeed_args="--master_port=11000"
gradient_checkpointing=True
conversation_template=llama2

# Other optional arguments that can improve memory saving
use_flash_attention=1

# Safety related arguments
trust_remote_code=0

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --conversation_template)
      conversation_template="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  contrib/tool-finetune/function_call_finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --conversation_template ${conversation_template} \
    --num_train_epochs 4 \
    --learning_rate 1e-6 \
    --disable_group_texts 1 \
    --block_size 2048 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing ${gradient_checkpointing} \
    --per_device_train_batch_size 4 \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --use_flash_attention ${use_flash_attention} \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_strategy steps\
    --save_steps 800 \
    --dataloader_num_workers 1 \
    > >(tee ${log_dir}/train.log) \
    2> >(tee ${log_dir}/train.err >&2)
