import json
import logging
import time
from datasets import load_dataset
from openai import OpenAI
import httpx
import re
from tqdm import tqdm


def extract_json_from_message(message):
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, message, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return message  # Return the original message if pattern not found

# Configure logging
logging.basicConfig(
    filename='lean_tool_generation.log',  # Log file name
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

# Initialize OpenAI client with custom API endpoint
client = OpenAI(
    base_url="https://www.apigptopen.xyz/v1", 
    api_key="sk-1y32BUDy6ZHG5Qvf3aBb2305C04f48F4Ae5f3727C9Ab0f6a",
    http_client=httpx.Client(
        base_url="https://www.apigptopen.xyz/v1",
        follow_redirects=True,
    ),
)

def process_data_item(data_item):
    global api_call_count, total_prompt_tokens, total_completion_tokens, total_tokens

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

    # Retry mechanism parameters
    max_retries = 10       # Maximum number of retries
    retry_delay = 60      # Delay between retries in seconds
    attempt = 0           # Current attempt count

    while attempt <= max_retries:
        try:
            # Use your client to call the API
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that provides responses in the specified JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Update global counters
            api_call_count += 1
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens_used = usage.total_tokens

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += total_tokens_used

            # Log API call details
            logging.info(f"API Call #{api_call_count}")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Completion: {completion.choices[0].message.content}")
            logging.info(f"Prompt Tokens: {prompt_tokens}")
            logging.info(f"Completion Tokens: {completion_tokens}")
            logging.info(f"Total Tokens: {total_tokens_used}")

            # Extract the generated text
            message = completion.choices[0].message.content.strip()
            # Parse the JSON output
            try:
                message_json = extract_json_from_message(message)
                
                def escape_unescaped_backslashes(s):
                    # Replace each backslash that is not already escaped
                    return re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', s)

                message_json = escape_unescaped_backslashes(message_json)
                output_json = json.loads(message_json)
                logging.info(f"JSON output for item '{name}': {output_json}")
                return output_json
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed for item '{name}': {e}")
                logging.error(f"JSON decoding failed for item '{name}': {e}")
                return None

        except Exception as e:
            attempt += 1
            logging.error(f"Error occurred for item '{name}' on attempt {attempt}: {e}")
            if attempt <= max_retries:
                logging.info(f"Retrying after {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded for item '{name}'. Skipping.")
                return None

# Load the dataset
dataset = load_dataset('RickyDeSkywalker/OpenBootstrappedTheorem')

# Process only the first 10 data items
tools = []

# Global counters to track API usage
api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# Get the first 10 data items for testing
data_items = dataset['train'].select(range(1000))
# Wrap the data_items iterator with tqdm for a progress bar
for data_item in tqdm(data_items, desc="Processing data items"):
    tool_info = process_data_item(data_item)
    if tool_info:
        tools.append(tool_info)

# Output API call statistics
print("\nAPI Call Statistics:")
print(f"Total API Calls: {api_call_count}")
print(f"Total Prompt Tokens: {total_prompt_tokens}")
print(f"Total Completion Tokens: {total_completion_tokens}")
print(f"Total Tokens Used: {total_tokens}")
# Log API call statistics
logging.info("API Call Statistics:")
logging.info(f"Total API Calls: {api_call_count}")
logging.info(f"Total Prompt Tokens: {total_prompt_tokens}")
logging.info(f"Total Completion Tokens: {total_completion_tokens}")
logging.info(f"Total Tokens Used: {total_tokens}")
logging.info("End of generate tool dataset.")
logging.info("--------------------------------------------------")

# Save the tools to a JSON file
with open('lean_tools.json', 'w', encoding='utf-8') as f:
    json.dump(tools, f, ensure_ascii=False, indent=4)
