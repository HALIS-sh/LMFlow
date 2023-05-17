import json
import torch
import os

from flask import Flask, request, stream_with_context
from flask import render_template
from flask_cors import CORS
from accelerate import Accelerator
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments


WINDOW_LENGTH = 512

# @dataclass
# class AppArguments:
#     end_string: Optional[str] = field(
#         default="##",
#         metadata={
#             "help": "end string mark of the chatbot's output"
#         },
#     )
#     max_new_tokens: Optional[int] = field(
#         default=200,
#         metadata={
#             "help": "maximum number of generated tokens"
#         },
#     )
# parser = HfArgumentParser((
#         ModelArguments,
#         AppArguments,
# ))

# model_args, pipeline_args, chatbot_args = (
#         parser.parse_args_into_dataclasses()
#     )

app = Flask(__name__)
CORS(app)
ds_config_path = "./examples/ds_config.json"
with open (ds_config_path, "r") as f:
    ds_config = json.load(f)


model_name_or_path = "/home/share/data/robin-33b"
# lora_path = '../output_models/instruction_ckpt/llama7b-lora/'
model_args = ModelArguments(model_name_or_path=model_name_or_path,torch_dtype="bfloat16")

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config, use_accelerator=True)
accelerator = Accelerator()


def stream_generate( inputs, context_len = 2048, max_new_tokens=128, *args, **kwargs):
    # input_ids = inputs
    # output_ids = input_ids

    max_src_len = context_len - 256 - 8
    # input_ids = input_ids[-max_src_len:]
    # print(len(input_ids))
    input_ids = model.tokenizer(inputs).input_ids

    input_echo_len = len(input_ids)
    print(input_echo_len)
    output_ids = list(input_ids)
    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None

    for i in range(0,512):
        if i == 0:
            with torch.no_grad():
                out = model.backend_model(torch.as_tensor([input_ids], device=local_rank), use_cache=True)
            logits = out.logits    
            past_key_values = out.past_key_values

        else:
            with torch.no_grad():
                out = model.backend_model(
                    input_ids=torch.as_tensor([[token]], device=local_rank),
                    use_cache=True,
                    past_key_values=past_key_values,
                )         
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0, -1, :]

        token = int(torch.argmax(last_token_logits))
        output_ids.append(token)

        tmp_output_ids = output_ids[input_echo_len:]
        rfind_start = 0

        output = model.tokenizer.decode(
            tmp_output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens = False,
        )
        # print(f"No. {i} A{output}A, length is {len(output)}， id is {tmp_output_ids}")
        # print("A" + output+  "A")
        yield output.replace("\ufffd","")


@app.route('/predict',methods = ['POST'])
def predict():
    if(request.method == "POST"):
        try:
            user_input = request.get_json()["Input"]
            conversation = request.get_json()["History"]

            history_input = ""
            if(len(conversation) >= 2):
                if(len(conversation) == 2):
                    history_input ="###Human: " + user_input +" "
                else:
                    for i in range(0, len(conversation)-1):
                        if(i % 2 == 0):
                            history_input = history_input + "###Human: "  + conversation[i+1]["content"] + " "
                        elif(i % 2 == 1):
                            history_input = history_input + "###Assistant:"  + conversation[i+1]["content"] 
                history_input = history_input +  "###Assistant:"

            if len(model.encode(history_input))> WINDOW_LENGTH:
                inputs = model.encode(history_input)
                inputs = inputs[-WINDOW_LENGTH:]
                history_input = model.decode(inputs)

            return app.response_class(stream_with_context(stream_generate(history_input)))

        except Exception as ex:
            print(ex)
            text_out = ex
    else:
        text_out = "Not POST Method"
    return text_out

@app.route('/',methods = ['GET'])
def login():

    return render_template('index.html')


app.run(port = 5000, debug = False)
