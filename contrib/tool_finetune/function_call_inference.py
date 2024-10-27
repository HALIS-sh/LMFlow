from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, DistributedSampler
import argparse, json, os, re
from tqdm import tqdm
import datetime
from typing import Optional, Dict
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Define tool function
class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    country: str = Field(description="One of the countries in the world, e.g. China, United States, United Kingdom")
import os
os.environ['NCCL_TIMEOUT'] = str(float('inf'))
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30),
)
def cleanup():
    dist.destroy_process_group()
def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--timeout", default=60, type=int)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    args = parser.parse_args()
    return args

def run(rank, world_size, test_question, model_path, temperature, top_p, max_tokens, batch_size, return_dict):
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return idx, self.data[idx]  # Return index along with data
        
    print(f"Process {rank} starting")
    setup(rank, world_size)
    print(f"Process {rank} finished setup")

    dataset = SimpleDataset(test_question)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    final_ans_jsons = []
    final_indices = []

    # test_question = [test_question[i : i + batch_size] for i in range(0, len(test_question), batch_size)]
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=f"cuda:{rank}", quantization_config=bnb_config)
    model = DDP(model, device_ids=[rank])
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    print(f"WARNING: OSSHandler is using customized model inference code from Shizhe on GPU {rank}!")

    for batch_indices, test_question_batch in tqdm(dataloader, disable=rank != 0):
        if rank == 0:
            print("start generating, test question length: ", len(test_question_batch))
            # print(f"test_question_batch: {test_question_batch}")
        model_inputs = tokenizer(test_question_batch, return_tensors="pt", padding=True).to(f"cuda:{rank}")
        input_ids = model_inputs.input_ids

        # Generate only new tokens
        generated_ids = model.module.generate(
            input_ids=input_ids,
            attention_mask=model_inputs.attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
        )
        generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        final_ans_jsons.extend(generated_texts)
        final_indices.extend(batch_indices.tolist())

    print(f"rank {rank} finished generating, final_ans_jsons length: {len(final_ans_jsons)}")

    # Gather results and indices from all GPUs
    all_final_ans_jsons = [None for _ in range(world_size)]
    all_final_indices = [None for _ in range(world_size)]
    dist.all_gather_object(all_final_ans_jsons, final_ans_jsons)
    dist.all_gather_object(all_final_indices, final_indices)

    if rank == 0:
        combined_final_ans_jsons = []
        combined_final_indices = []
        for ans_json, indices in zip(all_final_ans_jsons, all_final_indices):
            combined_final_ans_jsons.extend(ans_json)
            combined_final_indices.extend(indices)
        
        # Sort results based on original indices
        sorted_results = sorted(zip(combined_final_indices, combined_final_ans_jsons), key=lambda x: x[0])
        return_dict['result'] = [res for _, res in sorted_results]

    cleanup()

# Define function call parse
def parse_function_call(input_str: str) -> Optional[Dict[str, any]]:
    
    """
    Parses a text string to find and extract a function call.
    The function call is expected to be in the format:
    <functioncall> {"name": "<function_name>", "arguments": "<arguments_json_string>"}

    Args:
        input_str (str): The text containing the function call.

    Returns:
        Optional[Dict[str, any]]: A dictionary with 'name' and 'arguments' if a function call is found,
                                  otherwise None.
    """
    # Regex pattern to extract 'name' and 'arguments'
    pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*\{(.*?)\}'
    match = re.search(pattern, input_str)
    if match:
        try:
            name = match.group(1)
            arguments_str = "{" + match.group(2) + "}"
            arguments = json.loads(arguments_str)
            return {"name": name, "arguments": arguments}
        except json.JSONDecodeError:
            return None
    return None

class InferenceTool:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 0.0
        self.top_p = 1.0
        self.max_tokens = 200
        self.dtype = torch.float32
        self.model_name = model_path
        @tool("get_weather_tool", args_schema=SearchInput)
        def get_current_weather(location: str) -> str:
            """Get the current weather in a given location"""
            return {"location": location, "fahrenheit": 73.4}
        @tool("get_news_headlines_tool", args_schema=SearchInput)
        def get_news_headlines(country: str) -> list:
            """Get the latest news headlines"""
            if country == "United States":
                return {"headlines": ["Biden announces new vaccine mandates", "Hurricane Ida devastates Louisiana", "Apple unveils new iPhone", "NASA's Perseverance rover collects first Mars rock sample"]}
            else:
                return {"headlines": ["News headlines not available"]}

        print(dir(get_current_weather.func))  # View property list

        tools = [get_current_weather, get_news_headlines]
        self.functions = [convert_to_openai_function(t.func) for t in tools]

    def _format_prompt(prompt):
        if prompt['role'] == 'user':
            prompt_text = prompt['content']
            formatted_prompt = """<|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|>"""
        elif prompt['role'] == 'observation':
            prompt_text = str(prompt['content'])
            formatted_prompt = """<|start_header_id|>tool<|end_header_id|>\n\n{prompt_text}<|eot_id|>"""
        elif prompt['role'] == 'function':
            prompt_text = prompt['content']
            formatted_prompt = """<|start_header_id|>assistant<|end_header_id|>\n\n{prompt_text}"""
        else:
            assert False, "Prompt should start with user"

        # formatted_prompt = """<extra_id_0>System
        # You are a helpful assistant who has access to the following functions to help the user.
        # Your job is to solve the above question using ONLY and strictly ONE line of python code given the above functions. If you think no function should be invoked return "[]".
        # If you think one or more function should be invoked, return the function call in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)] wrapped in python code"\n
        # <extra_id_1>User
        # You can use the functions if needed-
        # <tool>{function}</tool>
        # Here is the question you need to answer:
        # {prompt_text}
        # <extra_id_1>Assistant\n"""
        # formatted_prompt = """<extra_id_0>System\nYou are a helpful assistant with access to the following functions. Use them if required -\n{function}\n<extra_id_1>User\n{prompt_text}\n<extra_id_1>Assistant\n"""
        # formatted_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant with access to the following functions. Use them if required -\n{function}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        # formatted_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant with access to the following functions. Use them if required -\n{function}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|>"""
        # formatted_prompt = """<extra_id_0>System\n\n<tool> {function} </tool>\n\n<extra_id_1>User\n{prompt_text}\n<extra_id_1>Assistant\n"""

        return formatted_prompt.format(prompt_text=prompt_text)

    def process_input(self, messages, format_prompt_func, include_system_prompt=True, model_name=None):
        prompts = []
        print('messages:', messages)
        for index, message in enumerate(messages):
            if index == 0 and include_system_prompt:
                functions = json.dumps(self.functions, indent=4)
                formated_system = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant with access to the following functions. Use them if required -\n{functions}<|eot_id|>"""
                prompts.append(formated_system)
                prompts.append(format_prompt_func(message))
            else:
                prompts.append(format_prompt_func(message))
        prompts.append("""<|start_header_id|>assistant<|end_header_id|>\n\n""")
        return prompts

    def chat_with_model(self, model, tokenizer, message, stop_token_ids=None, max_length=3000):
        # Ensure the model is on CUDA device (GPU)
        model = model.to("cuda")
        # Tokenize input messages and move tensors to the same device as model
        # model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        # Tokenize input messages and move tensors to the same device as model
        model_inputs = tokenizer(message,return_tensors="pt", padding=True).to("cuda")
        print("model_inputs:", model_inputs)
        # # Generate response using the model
        # outputs = model.generate(
        #     model_inputs, 
        #     max_length=max_length, 
        #     pad_token_id=tokenizer.eos_token_id,
        #     num_return_sequences=1,  # Ensures only one sequence is returned
        #     num_beams=1              # Disables beam search by setting it to 1
        # )
        input_ids = model_inputs.input_ids
        print("stop_token_ids:", stop_token_ids)
        # Generate only new tokens
        # generated_ids = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=model_inputs.attention_mask,
        #     do_sample=False,
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     max_new_tokens=self.max_tokens,
        #     stop_strings=["<|eot_id|>"],
        #     tokenizer=tokenizer  # Transfre tokenizer argument to the model
        # )
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=model_inputs.attention_mask,
            do_sample=False,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_tokens,
        )
        # generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        # Decode the output tensor to a string
        response = tokenizer.decode(generated_ids[:, input_ids.shape[1]:][0], skip_special_tokens=False)
        return response

    def inference(self, messages, num_gpus, gpu_memory_utilization, format_prompt_func=_format_prompt, stop_token_ids=None, max_model_len=None, include_system_prompt=True):
        # Process inputs
        processed_messages = self.process_input(messages, format_prompt_func, include_system_prompt=include_system_prompt, model_name=self.model_name)
        print('processes_messages', processed_messages)
        # # Perform a batch generate
        # ans_jsons = self._batch_generate(processed_messages=processed_messages, model_path=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, dtype=self.dtype, stop_token_ids=stop_token_ids, max_model_len=max_model_len, num_gpus=num_gpus, gpu_memory_utilization=gpu_memory_utilization)
        # # Generate the response
        # Let processed_messages[0] = the sum of the processed messages
        processed_messages[0] = ''.join(processed_messages)
        ans = self.chat_with_model(self.model,self.tokenizer, processed_messages[0],stop_token_ids=stop_token_ids)

        return ans, processed_messages

    @staticmethod
    def _batch_generate(processed_messages, model_path, temperature, max_tokens, top_p, dtype, stop_token_ids=None, max_model_len=None, num_gpus=3, gpu_memory_utilization=0.9):
        world_size = num_gpus
        batch_size = 1
        manager = mp.Manager()
        return_dict = manager.dict()

        torch.multiprocessing.spawn(run, args=(world_size, processed_messages, model_path, temperature, top_p, max_tokens, batch_size, return_dict), nprocs=world_size, join=True)

        final_ans_jsons = return_dict.get('result', None)
        if final_ans_jsons is not None:
            print("Length of Final result:", len(final_ans_jsons))
        else:
            print("No result returned")
        return final_ans_jsons


# Load local model and tokenizer
model_path = "/home/wenhesun/LMFlow/output_models/tool_finetuned_llama3_instruct_filtered_10000data_bs128/checkpoint-39"
# model_path = "/home/wenhesun/LMFlow/output_models/tool_finetuned_llama3_all_data_bs128"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
# Create InferenceTool Instance
tool_instance = InferenceTool(model, tokenizer)
SYSTEM_PROMPT = f"""You are a helpful assistant with access to the following functions. Use them if required - {json.dumps(tool_instance.functions, indent=4)}"""
print(SYSTEM_PROMPT)
args = get_args()
# Simulate the chat process
messages = [
    {'role': 'user', 'content': 'What is the weather like in Dallas, TX?'},
]
# messages = [
#     {'role': 'user', 'content': 'Can you tell me the latest news headlines for the United States?'},
# ]
# Use the local model to generate responses
stop_token_ids = [tokenizer.eos_token_id]
print("stop_token_ids:", stop_token_ids)
print("eos_token_ids", model.generation_config.eos_token_id)
results, processed_messages = tool_instance.inference(
    messages=messages,
    num_gpus=args.num_gpus,
    gpu_memory_utilization=args.gpu_memory_utilization,
    stop_token_ids=stop_token_ids,
)
print("Original results:", results)
messages.append({'role': 'function', 'content': results})

# # Analyze function calls
# function_call = parse_function_call(response)
# print("function_call:", function_call)
# if function_call and function_call.get("name") == "get_current_weather":
#     args = function_call.get("arguments")
#     print("args:", args)
#     messages.append({'role': 'observation', 'content': args})
#     current_weather = get_current_weather(**args)
#     messages.append({'role': 'assistant', 'content': 'Function Response: ' + str(current_weather)})
    
# messages.append({'role': 'observation', 'content': {'headlines': ["Biden announces new vaccine mandates", "Hurricane Ida devastates Louisiana", "Apple unveils new iPhone", "NASA's Perseverance rover collects first Mars rock sample"]}})
messages.append({'role': 'observation', 'content': {"location":"Dallas, TX" , "fahrenheit": 73.4}})
results, processed_messages = tool_instance.inference(
    messages=messages,
    num_gpus=args.num_gpus,
    gpu_memory_utilization=args.gpu_memory_utilization,
)
# Use the local model to generate responses
stop_token_ids = [tokenizer.eos_token_id]
# print("stop_token_ids:", stop_token_ids)
# Use local model to generate final response
results, processed_messages = tool_instance.inference(
    messages=messages,
    num_gpus=args.num_gpus,
    gpu_memory_utilization=args.gpu_memory_utilization,
    stop_token_ids=stop_token_ids,
)
print("Final results:", results)
messages.append({'role': 'assistant', 'content': results})
