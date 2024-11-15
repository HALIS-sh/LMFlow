import re, json, logging, time, os
from transformers import pipeline
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
import httpx

# Global counters to track API usage
api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# Initialize the LLM using vLLM
llm = LLM(model='/home/wenhesun/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693')  # Replace with your model path
# Initialize OpenAI client with custom API endpoint
# client = OpenAI(
#     base_url="https://www.apigptopen.xyz/v1", 
#     api_key="sk-1y32BUDy6ZHG5Qvf3aBb2305C04f48F4Ae5f3727C9Ab0f6a",
#     http_client=httpx.Client(
#         base_url="https://www.apigptopen.xyz/v1",
#         follow_redirects=True,
#     ),
# )

client = OpenAI(
    api_key="5aa04bb1-74ff-4042-b0a3-80826f8cb16f",
    base_url="https://api.sambanova.ai/v1",
)


# Configure logging
logging.basicConfig(
    filename='check_result_correctness_samba_3.log',  # Log file name
    filemode='w',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

def extract_last_boxed_content(solution_text):
    """
    From solution get the last \\boxed{...} content, support nested brackets.
    Args:
        solution_text (str): Contains the answer in LaTeX type.

    Returns:
        str: The content inside the last \\boxed{...}, or None if not found.
    """
    index = solution_text.rfind('\\boxed{')
    if index == -1:
        return None

    # From the position of '\\boxed{', find the matching brackets
    stack = ['{']
    i = index + len('\\boxed{')
    content = ''
    while i < len(solution_text):
        char = solution_text[i]
        if char == '{':
            stack.append('{')
            content += char
        elif char == '}':
            stack.pop()
            if not stack:
                # All {} are matched
                break
            else:
                content += char
        else:
            content += char
        i += 1

    if stack:
        # No matching closing curly brace found
        return None

    return content



def extract_python_output(python_output):
    '''
    Extract the required part from the python_output string based on the specified rules.
    - If python_output contains a colon ':', extract the part after the colon.
        - If the content contains mathematical symbols or variables, return the entire content.
        - Otherwise, if the content starts with a number, extract the number part.
    - If it does not contain a colon, try to extract the number from the beginning of the string.
    - If none of the above conditions are met, return the entire python_output.

    Args:
        python_output (str): The output string of Python code.
    
    Returns:
        str: The extracted required output part.
    '''
    try:
        python_output = python_output.strip()
    except:
        return python_output
    if ':' in python_output:
        # Get the part after the first colon
        parts = python_output.split(':', 1)
        result = parts[1].strip()
        # Check if the content contains mathematical symbols or variables
        if re.search(r'[a-zA-Z\^\*\+\-/\(\)]', result):
            return result
        else:
            # Check if the content starts with a number
            match = re.match(r'^[-+]?\d*\.?\d+', result)
            if match:
                number = match.group()
                return number
            else:
                return result
    else:
        # Check if the content starts with a number
        match = re.match(r'^[-+]?\d*\.?\d+', python_output)
        if match:
            number = match.group()
            return number
        else:
            return python_output

def extract_python_output_with_llm(python_output):
    # Retry mechanism parameters
    max_retries = 10       # Maximum number of retries
    retry_delay = 30      # Delay between retries in seconds
    attempt = 0           # Current attempt count

    chats = [
            {"role": "system", "content": "Your task is to help me extract only the number or the mathematical expression from the following sentence. You just need to answer me the number or the expression, no additional explanation is required."}
    ]
    chats.append({"role": "user", "content": f"The sentence is 'The expanded expression is: 5x^26 - 15x^9 + 5x^2 - 35x^4.'."})
    chats.append({"role": "assistant", "content": "5x^26 - 15x^9 + 5x^2 - 35x^4"})
    chats.append({"role": "user", "content": f"The sentence is 'The value of a * b is: 0.16.'."})
    chats.append({"role": "assistant", "content": "0.16"})
    chats.append({"role": "user", "content": f"The sentence is 'Sam did not work for 6 days.'."})
    chats.append({"role": "assistant", "content": "True"})
    chats.append({"role": "user", "content": f"The sentence is '360.00000000000006 seconds.'."})
    chats.append({"role": "assistant", "content": "360.00000000000006"})
    chats.append({"role": "user", "content": f"The sentence is 'The area of the rectangular garden is: 200.0 square feet.'."})
    chats.append({"role": "assistant", "content": "200.0"})
    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model='Meta-Llama-3.1-405B-Instruct',
                messages=[{"role": "user", "content": f"The sentence is {python_output}"}],
                temperature =  0.1,
                top_p = 0.1
            )
            message = response.choices[0].message.content
            return message
        except Exception as e:
            attempt += 1
            logging.error(f"Error occurred on attempt {attempt}: {e}")
            if attempt <= max_retries:
                logging.info(f"Retrying after {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded. Skipping.")
                return None

def generate_answer_with_gpt(conversation):
    """
    Generates an answer using the GPT model.

    Args:
        conversation (list): A list of conversation messages.

    Returns:
        str: The generated answer.
    """
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
                messages=conversation,
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
            logging.info(f"Completion: {completion.choices[0].message.content}")
            logging.info(f"Prompt Tokens: {prompt_tokens}")
            logging.info(f"Completion Tokens: {completion_tokens}")
            logging.info(f"Total Tokens: {total_tokens_used}")

            # Extract the generated text
            message = completion.choices[0].message.content.strip()
            return message
        except Exception as e:
            attempt += 1
            logging.error(f"Error occurred for item '{name}' on attempt {attempt}: {e}")
            if attempt <= max_retries:
                logging.info(f"Retrying after {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded for item '{name}'. Skipping.")
                return None
        
def generate_answer_with_samba(conversation, idx, max_new_tokens=256):

    # Retry mechanism parameters
    max_retries = 10       # Maximum number of retries
    retry_delay = 10      # Delay between retries in seconds
    attempt = 0           # Current attempt count

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model='Meta-Llama-3.1-405B-Instruct',
                messages=conversation,
                temperature =  0.1,
                top_p = 0.1
            )
            message = response.choices[0].message.content
            return message
        except Exception as e:
            attempt += 1
            logging.error(f"Error occurred for item{idx} on attempt {attempt}: {e}")
            if attempt <= max_retries:
                logging.info(f"Retrying after {retry_delay} seconds... (Attempt {attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded for item{idx}. Skipping.")
                return None
      

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

def verify_solution_with_model(solution_text, expected_answer):
    """
    Uses an open-source model to verify if the solution leads to the expected answer.

    Args:
        solution_text (str): The solution text.
        expected_answer (float): The numerical answer extracted from the solution.

    Returns:
        bool: True if the model confirms the solution is correct, False otherwise.
    """
    # Initialize a question-answering pipeline with an open-source model
    nlp = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

    # Prepare the question and context
    question = "What is the final answer?"
    context = solution_text

    # Get the model's answer
    result = nlp(question=question, context=context)

    # Extract the numerical answer from the model's output
    try:
        model_answer = float(result['answer'])
        return abs(model_answer - expected_answer) < 1e-6  # Allowing a small margin for float comparison
    except:
        return False

# # Example data
# solution = r"We note that $f(-2)=(-2)^3+3=-5$, so $g(f(-2))=g(-5)=2\cdot(-5)^2+2\cdot(-5)+1=41.$ Therefore our answer is $\boxed{41}$."
# python_output = "The result is: 41"


# Initialize the file path
file_path = '/home/wenhesun/LMFlow/contrib/prepare_data/math_low_level_algebra_python.jsonl'

# Initialize the results list
results = []

# Open the file to count the total number of lines
with open(file_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# Open the file to process the data
with open(file_path, 'r', encoding='utf-8') as f:
    true_count = 0
    false_count = 0
    for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing progress"), 1):
        data = json.loads(line.strip())
        solution = data.get('solution', '')
        python_output = data.get('python_output', '')
        boxed_content = extract_last_boxed_content(solution)
        python_result = extract_python_output_with_llm(python_output)
        chats = [
                {"role": "system", "content": "You are a mathematician and proficient in LateX and Python. Your task is to help me determine if the results I gave in python of type String and type Latex type have the same meaning. You just need to answer me True or False, no additional explanation is required."}
        ]
        chats.append({"role": "user", "content": f"whether the expression of (5x^26 - 15x^9 + 5x^2 - 35x^4) and the reult of latex (5x^{26}-15x^9-35x^4+5x^2) have the same meaning."})
        chats.append({"role": "assistant", "content": "True"})
        chats.append({"role": "user", "content": f"whether the expression of ((3.0y - -5.0)^2) and the reult of latex ((3y - 5)^2) have the same meaning."})
        chats.append({"role": "assistant", "content": "True"})
        chats.append({"role": "user", "content": f"whether the expression of (140.0) and the reult of latex (140) have the same meaning."})
        chats.append({"role": "assistant", "content": "True"})
        chats.append(
            {"role": "user", "content": f"whether the expression of ({python_result}) and the reult of latex ({boxed_content}) have the same meaning."})

        # Generate the answer
        # generated_text = generate_answer(chats)
        generated_text = generate_answer_with_samba(chats, line_num)
        if generated_text.find('True'):
            true_count += 1
        elif generated_text.find('False'):
            false_count += 1
        else:
            logging.info(f"Error: {generated_text}")
        print(generated_text)
        # Log API call details
        logging.info(f"Check_result: #{generated_text}")
        logging.info(f"Python_output: {python_output}")
        logging.info(f"Solution: {solution}")
        logging.info(f"Python_result: {python_result}")
        logging.info(f"Boxed_content: {boxed_content}")
        logging.info("--------------------------------------------------")
logging.info(f"True count: {true_count}")
logging.info(f"False count: {false_count}")
logging.info(f"Total count: {total_lines}")
logging.info(f"True rate: {true_count/total_lines}")
logging.info(f"False rate: {false_count/total_lines}")

# # Extract answers
# expected_answer = extract_answer_from_solution(solution)
# computed_result = extract_result_from_output(python_output)

# # Verify the solution with the model (optional)
# is_solution_correct = verify_solution_with_model(solution, expected_answer)

# # Compare the results
# if expected_answer is not None and computed_result is not None:
#     if abs(expected_answer - computed_result) < 1e-6 and is_solution_correct:
#         print("The python output is consistent with the solution.")
#     else:
#         print("The python output is NOT consistent with the solution.")
# else:
#     print("Could not extract numerical results from the data.")
