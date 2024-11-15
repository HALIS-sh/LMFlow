from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
# Initialize the LLM using vLLM
# llm = LLM(model='/home/wenhesun/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693', tensor_parallel_size = 2)  # Replace with your model path
llm = LLM(model='/home/wenhesun/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac',tensor_parallel_size = 2, gpu_memory_utilization = 0.9)  # Replace with your model path
def generate_answer(conversation, max_new_tokens=256):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=None,  # No stop tokens
    )
    outputs = llm.chat(conversation, sampling_params=sampling_params)
    output = outputs[0]
    generated_text = output.outputs[0].text
    # Uncomment the next line if you want to count the number of generated tokens
    # num_generated_tokens = output.outputs[0].num_tokens
    return generated_text  # , num_generated_tokens




def process_data_item(data_item):
    # Extract the relevant information
    name = data_item['Name']
    generated_text = data_item['Generated_informal_statement_and_proof']
    proof = data_item['Proof']
    # Create a prompt for the API
    prompt = f'''
    You are provided with a theorem and its informal statement and proof:

    {generated_text}

    Based on this information:
    - Extract the conditions (preconditions) necessary for the theorem.
    - Identify the conclusions (postconditions) that result from the theorem.
    - Generate a brief description of a tool that encapsulates this theorem.
    - Format the output as a JSON object with the following structure:

    {{
        "name": "{name}",
        "description": "A brief description of the tool.",
        "conditions": ["Condition1", "Condition2"],
        "conclusions": ["Conclusion1"],
        "function_body": "{proof}"
    }}

    Ensure the JSON is properly formatted.
    '''

    messages=[
        {"role": "system", "content": "You are an AI assistant that provides responses in the specified JSON format."},
        {"role": "user", "content": prompt}
    ]
    generated_answer = generate_answer(messages)
    return generated_answer



# Load the dataset
dataset = load_dataset('RickyDeSkywalker/OpenBootstrappedTheorem')

# Process only the first 10 data items
tools = []
# Get the first 10 data items for testing
data_items = dataset['train'].select(range(10))
# data_items = dataset['train'].select(range(1000, 10000))
# Wrap the data_items iterator with tqdm for a progress bar
for data_item in tqdm(data_items, desc="Processing data items"):
    tool_info = process_data_item(data_item)
    if tool_info:
        tools.append(tool_info)