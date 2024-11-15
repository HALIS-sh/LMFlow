import json
import re
import sys, os
import logging
from typing import List, Dict
from openai import OpenAI
import httpx
import time

# Configure logging
logging.basicConfig(
    filename='math_geometry_generation.log',  # Log file name
    filemode='w',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

# Initialize OpenAI client with custom API endpoint
client = OpenAI(
    base_url="https://www.apigptopen.xyz/v1", 
    api_key="xxx",
    http_client=httpx.Client(
        base_url="https://www.apigptopen.xyz/v1",
        follow_redirects=True,
    ),
)

api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0
# Get the project root directory
project_root = os.path.abspath(os.path.dirname(__file__))
# Add the project root directory to the system path
sys.path.append(project_root)
from prompt import MESSAGE_PROMPT, EXAMPLE

def extract_function_details(code: str) -> List[Dict[str, str]]:
    # 提取import语句
    import_pattern = re.compile(r'^\s*(import\s+\S+(\s+as\s+\w+)?)|(^\s*from\s+\S+\s+import\s+\S+(\s+as\s+\w+)?)', re.MULTILINE)
    imports = [match.group(0).strip() for match in import_pattern.finditer(code)]

    # 正则表达式模式匹配函数名称、描述、参数和返回值
    pattern = re.compile(
        r'(def\s+(?P<name>\w+)\s*\(.*?\):\s*)'  # 匹配函数头
        r'("""(?P<docstring>.*?)"""|\'\'\'(?P<docstring_single>.*?)\'\'\')'  # 匹配docstring，支持双引号和单引号
        r'(?P<body>.*?)\n(?=\n|def|$)',  # 匹配函数体
        re.DOTALL
    )

    functions = []
    for match in pattern.finditer(code):
        # 优先使用双引号匹配的docstring，如果没有则使用单引号
        docstring = match.group("docstring") or match.group("docstring_single")

        # 提取 description，Args 和 Returns 部分
        description_match = re.search(r'^(.*?)\n\s*Args:', docstring, re.DOTALL)
        args_match = re.search(r'Args:\s*(.*?)\n\s*Returns:', docstring, re.DOTALL)
        returns_match = re.search(r'Returns:\s*(.*)', docstring, re.DOTALL)

        description = description_match.group(1).strip() if description_match else ""
        args = [line.strip() for line in args_match.group(1).strip().splitlines() if line.strip()] if args_match else []
        returns = [line.strip() for line in returns_match.group(1).strip().splitlines() if
                   line.strip()] if returns_match else []

        # 获取函数头和去掉docstring的函数体
        function_header = match.group(1).strip()
        function_body = match.group("body").strip()
        full_function_body = f"{function_header}\n{function_body}"

        function_details = {
            "name": match.group("name"),
            "description": description,
            "conditions": args,
            "conclusions": returns,
            "requirement": imports,
            "function_body": full_function_body
        }
        functions.append(function_details)
    return functions

def call_openai(prompt: str, idx: int) -> str:
    global api_call_count, total_prompt_tokens, total_completion_tokens, total_tokens

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
            return message
        except Exception as e:
            attempt += 1
            logging.error(f"Error occurred for item '{idx}' on attempt {attempt}: {e}")
            if attempt <= max_retries:
                logging.info(f"Retrying after {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded for item '{idx}'. Skipping.")
                return None

# subset = 'geometry'
subset = 'test'
data = []
with open(f'math_low_level_{subset}_python.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

idx = 0
for entry in data:
    print('Generating index ', idx)
    code = entry['python']
    try:
        # 生成 function tools
        function_details = json.dumps(extract_function_details(code), indent=4)
        entry['tools'] = function_details

        prompt = MESSAGE_PROMPT.format(example=EXAMPLE,
                                       problem=entry['problem'],
                                       solution=entry['solution'],
                                       python_solution=entry['python'])
        response = call_openai(prompt, idx)
        json_res = response.split('```json')[1].split('```')[0]
        python_res = response.split('```python')[1].split('```')[0] if '```python' in response else None
        entry['messages'] = json.loads(json_res)
        if python_res:
            entry['missing_tools'] = json.dumps(extract_function_details(python_res), indent=4)
    except Exception as e:
        entry['error'] = e
    idx += 1

with open(f'math_low_level_{subset}_4o.jsonl', 'w', encoding='utf-8') as f:
    for entry in data:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + '\n')
