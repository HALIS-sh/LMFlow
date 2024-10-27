from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from typing import Optional, Dict
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from vllm import SamplingParams, LLM

# Define tool function
class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

# @tool("get_weather_tool", args_schema=SearchInput)
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    return {"location": location, "fahrenheit": 73.4}

# print(dir(get_current_weather.func))  # View property list

tools = [get_current_weather]
functions = [convert_to_openai_function(t) for t in tools]
SYSTEM_PROMPT = f"""You are a weather assistant with access to these functions -{json.dumps(functions, indent=4)}"""
print(SYSTEM_PROMPT)

# Load local model and tokenizer
model_path = "/home/wenhesun/LMFlow/output_models/tool_finetuned_llama3_instruct_filtered_10000data_bs128"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Define the chat function
# def chat_with_model(model, tokenizer, history, max_length=3000):
#     input_text = " ".join(history)
#     inputs = tokenizer(input_text, return_tensors="pt")
#     outputs = model.generate(inputs['input_ids'], max_length=max_length, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response
def chat_with_model(model, tokenizer, messages, max_length=3000):
    # Ensure the model is on CUDA device (GPU)
    model = model.to("cuda")
    # Tokenize input messages and move tensors to the same device as model
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    # Generate response using the model
    outputs = model.generate(
        input_ids=input_ids, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,  # Ensures only one sequence is returned
        num_beams=1              # Disables beam search by setting it to 1
    )
    # Decode the output tensor to a string
    response = tokenizer.decode(outputs[:, input_ids.shape[1]:][0], skip_special_tokens=True)
    return response

def chat_with_vllm(model_path, history):

    sampling_params = SamplingParams(
        use_beam_search=False,
        n=1,
        temperature=1e-6,
        max_tokens=3000,
        seed=1,
        top_p=1.0,
        top_k=1,
    )
    # Create an LLM.
    llm = LLM(model=model_path)
    vllm_outputs = llm.generate(
        history,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

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
    input_str = input_str.replace('\n', '')
    # print("input_str:", input_str)
    match = re.search(pattern, input_str)
    # print("match:", match)
    if match:
        try:
            name = match.group(1)
            arguments_str = "{" + match.group(2) + "}"
            arguments = json.loads(arguments_str)
            return {"name": name, "arguments": arguments}
        except json.JSONDecodeError:
            return None
    return None

# Simulate the process of chat
messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT}, 
    {'role': 'user', 'content': 'What is the weather like in Dallas, TX?'},
]

# Use local model to generate the response
history = [msg['content'] for msg in messages]
response = chat_with_model(model, tokenizer, messages)
print("Original response:", response)
print("--------------original response end--------------")
messages.append({'role': 'function', 'content': response})
# Analysis the function call
function_call = parse_function_call(response)
# print("function_call:", function_call)
if function_call and function_call.get("name") == "get_current_weather":
    args = function_call.get("arguments")
    # print("args:", args)
    function_response = get_current_weather(**args)
    messages.append({'role': 'observation', 'content': str(function_response)})

# Use local model to generate the Final response
# history = [msg['content'] for msg in messages]
final_response = chat_with_model(model, tokenizer, messages)
print("Final response:", final_response)
print("--------------final response end--------------")
messages.append({'role': 'assistant', 'content': final_response})