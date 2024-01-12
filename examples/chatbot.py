#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import warnings

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.utils.vllm import LLM, SamplingParams


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )

def main():
    pipeline_name = "inferencer"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        PipelineArguments,
        ChatbotArguments,
    ))
    model_args, pipeline_args, chatbot_args = (
        parser.parse_args_into_dataclasses()
    )
    inferencer_args = pipeline_args

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = AutoModel.get_model(
        model_args,
        tune_strategy='none',
        ds_config=ds_config,
        device=pipeline_args.device,
    )

    # We don't need input data, we will read interactively from stdin
    data_args = DatasetArguments(dataset_path=None)
    dataset = Dataset(data_args)

    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"
    # #add_code
    # vllm_params = SamplingParams(temperature=inferencer_args.temperature, top_p=1)
    # # Create an LLM.
    # llm = LLM(model=model_args.model_name_or_path)
    # # add_code_end
    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message)

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""
    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure
    use_vllm_flag = input("Please choose whether use vllm service(True/False):")
    try:
        use_vllm_flag = bool(eval(use_vllm_flag))
    except (NameError, SyntaxError):
        print("输入无效，请输入 True 或 False")
        # 如果用户输入无法解析为 bool 值，这里可以设置一个默认值，或者进行其他处理
        use_vllm_flag = False 

    while True:
        input_text = input("User >>> ")
        if input_text == "exit":
            print("exit...")
            break
        elif input_text == "reset":
            context = ""
            print("Chat history cleared")
            continue
        if not input_text:
            input_text = " "

        context += prompt_structure.format(input_text=input_text)
        context = context[-model.get_max_length():]     # Memory of the bot

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": context } ]
        })

        print("Bot: ", end="")
        print_index = 0

        token_per_step = inferencer_args.max_new_tokens

        for response, flag_break in inferencer.stream_inference(
            context=context,
            model=model,
            max_new_tokens=inferencer_args.max_new_tokens,
            num_beams=inferencer_args.num_beams,
            top_p=inferencer_args.top_p,
            token_per_step=token_per_step,
            temperature=inferencer_args.temperature,
            end_string=end_string,
            input_dataset=input_dataset,
            use_vllm_flag = use_vllm_flag
        ):
            # Prints characters in the buffer
            new_print_index = print_index
            for char in response[print_index:]:
                if end_string is not None and char == end_string[0]:
                    if new_print_index + len(end_string) >= len(response):
                        break

                new_print_index += 1
                print(char, end="", flush=True)

            print_index = new_print_index

            if flag_break:
                break
        print("\n", end="")

        context += response + "\n"


if __name__ == "__main__":
    main()
