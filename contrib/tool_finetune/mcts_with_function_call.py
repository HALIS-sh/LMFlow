import logging

# Configure logging
logging.basicConfig(
    filename='mcts_gpt_integration_with_tools.log',  # Log file name
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

# Control the log level of third packages
logging.getLogger("bitsandbytes").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import json
import math
import time
import random
import re
import os
from functools import lru_cache
import requests
from typing import Any, Dict, List
from openai import OpenAI
import httpx
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path to your tools JSON file
TOOLS_JSON_PATH = '/home/wenhesun/LMFlow/contrib/prepare_data/unique_extracted_tools.json'
EMBEDDINGS_PATH = '/home/wenhesun/LMFlow/contrib/prepare_data/python_tools_embeddings.npy'

# Initialize the sentence transformer model
model = SentenceTransformer('/home/wenhesun/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a')
tool_embeddings = {}

# Load tools from JSON
with open(TOOLS_JSON_PATH, 'r') as file:
    tools = json.load(file)

# Check if embeddings are already computed and saved
if os.path.exists(EMBEDDINGS_PATH):
    embeddings_matrix = np.load(EMBEDDINGS_PATH)
    tool_ids = list(tool['name'] for tool in tools)
    logging.info(f"Loaded tool embeddings from {EMBEDDINGS_PATH}")
else:
    # Precompute embeddings for each tool's description
    tool_embeddings = [model.encode(tool['name'] + tool['description'], convert_to_tensor=False) for tool in tools]
    embeddings_matrix = np.array(tool_embeddings)
    tool_ids = list(tool['name'] for tool in tools)
    np.save(EMBEDDINGS_PATH, embeddings_matrix)
    logging.info(f"Computed and saved tool embeddings to {EMBEDDINGS_PATH}")

# ----------------------------
# 1. Configuration and Setup
# ----------------------------

# Initialize OpenAI client with custom API endpoint
client = OpenAI(
    base_url="https://www.apigptopen.xyz/v1", 
    api_key="xxx",  # Replace with your actual API key
    http_client=httpx.Client(
        base_url="https://www.apigptopen.xyz/v1",
        follow_redirects=True,
    ),
)

# Configuration for Azure OpenAI
API_KEY = os.environ.get("AZUREAI_API_KEY")
ENDPOINT = os.environ.get("AZUREAI_ENDPOINT_URL")

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Global counters to track API usage
api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# ----------------------------
# 2. Helper Functions
# ----------------------------
def gpt_generate_use_azure(prompt: str, model: str = 'gpt-4', max_tokens: int = 150) -> str:
    """
    Calls the GPT-4 API via Azure OpenAI to generate a response and logs relevant information.
    """
    global api_call_count, total_prompt_tokens, total_completion_tokens, total_tokens
    # Retry mechanism parameters
    max_retries = 10       # Maximum number of retries
    retry_delay = 30      # Delay between retries in seconds
    attempt = 0           # Current attempt count
    while attempt <= max_retries:
        try:
            # Payload for the request
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that provides responses in the specified JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "top_p": 0.95,
                "max_tokens": max_tokens
            }
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_data = response.json()
            # Update global counters
            api_call_count += 1
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens_used = usage.get('total_tokens', 0)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += total_tokens_used

            # Log API call details
            logging.info(f"API Call #{api_call_count}")
            logging.info(f"Prompt: {prompt}")
            if 'choices' in response_data and len(response_data['choices']) > 0:
                completion_content = response_data['choices'][0]['message']['content']
                logging.info(f"Completion: {completion_content}")
                message = completion_content.strip()
            else:
                logging.error("No choices found in the response.")
                return ""

            logging.info(f"Prompt Tokens: {prompt_tokens}")
            logging.info(f"Completion Tokens: {completion_tokens}")
            logging.info(f"Total Tokens: {total_tokens_used}")

            return message
        except requests.RequestException as e:
            attempt += 1
            if attempt <= max_retries:
                logging.warning(f"API request failed. Retrying in {retry_delay} seconds. Attempt {attempt}/{max_retries}")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries exceeded. Failed to make the request. Error: {e}")
                return ""
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response. Error: {e}")
            return ""


def extract_json(response: str) -> Dict[str, Any]:
    """
    Extracts JSON block from the GPT response.
    Assumes JSON is enclosed within curly braces.
    """
    try:
        # Use regex to find content within curly braces
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
    return {}

def parse_initial_question(question: str) -> Dict[str, Any]:
    """
    Parses the initial user question into the required JSON format.
    
    Expected JSON format:
    {
        "initial_conditions": ["Condition1", "Condition2", ...],
        "initial_conclusions": ["Conclusion1", "Conclusion2", ...],
        "goal_conclusions": {"Goal1", "Goal2", ...}
    }
    """
    prompt = f"""
    Given the user's initial question, parse it into the following JSON format:

    {{
        "initial_conditions": ["..."],
        "initial_conclusions": [...],
        "goal_conclusions": {{}}
    }}

    Initial Question:
    "{question}"

    Only provide the JSON output.
    """

    response = gpt_generate_use_azure(prompt)
    logging.info(f"GPT Response for Initial Question Parsing: {response}")
    data = extract_json(response)
    return data

# ----------------------------
# 3. Core Classes
# ----------------------------

@lru_cache(maxsize=1024)
def is_goal_state_gpt_cached(current_conclusions_json: str, goal_conclusions_json: str) -> bool:
    """
    Cached version of is_goal_state_gpt using JSON string representations.
    
    Args:
        current_conclusions_json (str): JSON string of current conclusions.
        goal_conclusions_json (str): JSON string of goal conclusions.
    
    Returns:
        bool: True if the goal is achieved, False otherwise.
    """
    # Deserialize JSON strings to Python objects
    current_conclusions = json.loads(current_conclusions_json)
    goal_conclusions = set(json.loads(goal_conclusions_json))
    
    # Construct the prompt for GPT-4
#     prompt = f"""
# Determine if the following goal conclusions have been achieved based on the current conclusions.

# Current Conclusions:
# {json.dumps(current_conclusions, indent=2)}

# Goal Conclusions:
# {json.dumps(list(goal_conclusions), indent=2)}

# Answer with 'True' if all goal conclusions are achieved, otherwise 'False'. Do not provide any additional text.
# """
    prompt = f"""
Determine if the following goal conclusions have been achieved based on the current conclusions. Analyze the content semantically to assess whether each goal conclusion is satisfied.

Current Conclusions:
{json.dumps(current_conclusions, indent=2)}

Goal Conclusions:
{json.dumps(list(goal_conclusions), indent=2)}

Answer with 'True' if all goal conclusions are achieved, otherwise 'False'. Answer "True" or "False" with the explanation. Your answer must contain the words "True" or "False".
"""


    # Call GPT-4 to evaluate
    response = gpt_generate_use_azure(prompt)
    logging.info(f"GPT Response for Goal State Check: {response}")
    
    # Process GPT response
    response_clean = response.strip().lower()
    if 'true' in response_clean:
        return True
    elif 'false' in response_clean:
        return False
    else:
        logging.warning(f"Unexpected GPT response for goal state check: '{response}'. Defaulting to False.")
        return False


class State:
    """
    Represents the current state, including conditions, conclusions, and messages.
    """
    def __init__(self, conditions: List[str], conclusions: List[str], messages: List[Dict[str, Any]] = None):
        self.conditions = set(conditions)
        self.conclusions = set(conclusions)
        self.messages = messages if messages is not None else []

    # def is_goal_state(self, goal_conclusions: set) -> bool:
    #     """
    #     Determines whether the goal has been achieved based on the state's conclusions.
    #     """
    #     return goal_conclusions.issubset(self.conclusions)
    
    def is_goal_state_gpt(self, goal_conclusions: set) -> bool:
        """
        Determines whether the goal has been achieved based on the current conclusions using GPT-4.
        
        Args:
            current_conclusions (List[str]): The list of current conclusions.
            goal_conclusions (Set[str]): The set of goal conclusions.
        
        Returns:
            bool: True if the goal is achieved, False otherwise.
        """
        # Serialize the inputs to JSON strings for caching
        current_conclusions_json = json.dumps(list(self.conclusions), sort_keys=True)
        goal_conclusions_json = json.dumps(list(goal_conclusions), sort_keys=True)
        
        # Call the cached helper function
        return is_goal_state_gpt_cached(current_conclusions_json, goal_conclusions_json)



    def __eq__(self, other):
        return (
            self.conditions == other.conditions and
            self.conclusions == other.conclusions and
            self.messages == other.messages
        )

    def __hash__(self):
        return hash((
            frozenset(self.conditions),
            frozenset(self.conclusions),
            tuple(tuple(sorted(msg.items())) for msg in self.messages)
        ))

    def __str__(self):
        return f"Conditions: {', '.join(self.conditions)}; Conclusions: {', '.join(self.conclusions)}"

class Node:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state: State, parent: 'Node' = None, action: str = None):
        self.state = state         # Current state
        self.parent = parent       # Parent node
        self.action = action       # Action (tool application) from parent to current node
        self.children = []         # List of child nodes
        self.visits = 0            # Number of times node has been visited
        self.reward = 0            # Accumulated reward

# ----------------------------
# 4. Tool Definitions
# ----------------------------

# # Define the list of tools
# tools = [
#     {
#         "name": "absolute_value",
#         "description": "Calculate the absolute value of a number.",
#         "conditions": [
#             "number (int or float): The number to calculate the absolute value of."
#         ],
#         "conclusions": [
#             "int or float: The absolute value of the input number."
#         ],
#         "requirement": [
#             "import math"
#         ],
#         "function_body": "def absolute_value(number):\n    return abs(number)"
#     },
#     {
#         "name": "add_vectors",
#         "description": "Add two 3-dimensional vectors.",
#         "conditions": [
#             "vector_a (array-like): First vector with 3 components (x, y, z).",
#             "vector_b (array-like): Second vector with 3 components (x, y, z)."
#         ],
#         "conclusions": [
#             "np.ndarray: The result of adding vector_a and vector_b."
#         ],
#         "requirement": [
#             "import numpy as np"
#         ],
#         "function_body": "def add_vectors(vector_a, vector_b):\n    return np.array(vector_a) + np.array(vector_b)"
#     },
#     {
#         "name": "get_news_headlines",
#         "description": "Get the latest news headlines",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "country": {
#                     "type": "string",
#                     "description": "The country for which to fetch news"
#                 }
#             },
#             "required": ["country"]
#         },
#         "conditions": [
#             "Country name is provided."
#         ],
#         "conclusions": [
#             "Latest news headlines for the specified country are obtained."
#         ],
#         "function_body": """
# def get_news_headlines(country):
#     headlines = [
#         f"Breaking news in {country}: Headline 1",
#         f"Latest update in {country}: Headline 2",
#         f"Top story in {country}: Headline 3",
#         f"News flash in {country}: Headline 4"
#     ]
#     return {"headlines": headlines}
# """
#     },
#     # Add more tools as needed
# ]

# ----------------------------
# 5. Core Functions
# ----------------------------


def execute_tool_function(requirements: List[str], function_body: str, args: Dict[str, Any]) -> Any:
    """
    Executes the provided function_body as Python code with the given arguments.
    First executes the requirements to set up the environment.
    
    Args:
        requirements (List[str]): List of Python code strings to execute before the function.
        function_body (str): The Python function code.
        args (dict): The arguments to pass to the function.
    
    Returns:
        Any: The result of the function execution.
    """
    # Use a share name space
    env = {}
    
    # Execute each requirement
    for req in requirements:
        try:
            exec(req, env, env)
            logging.info(f"Executed requirement: {req}")
        except Exception as e:
            logging.error(f"Error executing requirement '{req}': {e}")
            # Re-raise exception to be caught in the calling function
            raise e
    
    # Execute the function body
    try:
        exec(function_body, env, env)
        logging.info(f"Executed function body: {function_body}")
    except Exception as e:
        logging.error(f"Error executing function body: {e}")
        # Re-raise exception to be caught in the calling function
        raise e
    
    # Extract the function name using regex
    func_name_match = re.search(r"def (\w+)\(", function_body)
    if func_name_match:
        func_name = func_name_match.group(1)
    else:
        error_msg = "Function name could not be determined from function_body."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if the function exists in env
    if func_name in env:
        try:
            result = env[func_name](**args)
            logging.info(f"Function '{func_name}' executed successfully with arguments {args}. Result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error executing function '{func_name}' with arguments {args}: {e}")
            # Re-raise exception to be caught in the calling function
            raise e
    else:
        error_msg = f"Function '{func_name}' not found after execution."
        logging.error(error_msg)
        raise ValueError(error_msg)



# def get_possible_actions(state: State, goal_conclusions: set) -> List[str]:
#     """
#     Calls GPT-4 to get the next possible tools based on the current state and goal conclusions.
#     Follows Steps 1 and 2:
#     Step 1: Determine next possible tools by calling GPT.
#     Step 2: Format the GPT prompt as the content of role user in the message.
#     """
#     # Step 1: Create the prompt for GPT to suggest next possible tools
#     prompt = f"""
# Based on the current state and goal, please suggest possible next tools to apply. The response must be in the following JSON format:

# {{
#     "current_conditions": {json.dumps(list(state.conditions))},
#     "current_conclusions": {json.dumps(list(state.conclusions))},
#     "goal_conclusions": {json.dumps(list(goal_conclusions))},
#     "next_possible_tools": ["Tool1", "Tool2", ...]
# }}

# Use the following tool descriptions to determine applicable tools:

# """
#     for tool in tools:
#         prompt += f"Tool Name: {tool['name']}\nDescription: {tool['description']}\nConditions: {', '.join(tool['conditions'])}\nConclusions: {', '.join(tool['conclusions'])}\n\n"

#     # Step 2: Add the prompt as a user message to the state's messages
#     user_message = {
#         "role": "user",
#         "content": prompt.strip()
#     }
#     state.messages.append(user_message)
#     logging.info(f"Added user message to state: {user_message}")

#     # Call GPT to get the next possible tools
#     response = gpt_generate_use_azure(prompt)
#     logging.info(f"GPT Response for Possible Actions: {response}")

#     # Extract JSON from the response
#     data = extract_json(response)
#     actions = data.get('next_possible_tools', [])

#     # Filter actions to ensure they are valid tool names
#     valid_actions = [action for action in actions if action in [tool['name'] for tool in tools]]
#     logging.info(f"Next possible tools after filtering: {valid_actions}")
#     return valid_actions

def get_possible_actions(state: State, goal_conclusions: set, top_n: int = 10) -> List[str]:
    """
    Retrieves the top N most relevant tools based on the current state and goal conclusions.
    
    Parameters:
    - state (State): The current state containing conditions, conclusions, and messages.
    - goal_conclusions (set): The set of goal conclusions.
    - top_n (int): The number of top tools to retrieve.
    
    Returns:
    - List[str]: A list of tool names that are most relevant.
    """
    # Step 1: Prepare the current state text
    current_text = " ".join(state.conditions) + " " + " ".join(state.conclusions)
    
    # Step 2: Compute embedding for the current state
    current_embedding = model.encode(current_text, convert_to_tensor=False)
    
    # Step 3: Compute cosine similarity between current state and all tools
    similarities = cosine_similarity([current_embedding], embeddings_matrix)[0]
    
    # Step 4: Get indices of top N similar tools
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Step 5: Retrieve corresponding tool names
    top_tool_ids = [tool_ids[idx] for idx in top_indices]
    
    # Optional: Log the retrieved tools and their similarity scores
    for idx in top_indices:
        tool_id = tool_ids[idx]
        similarity = similarities[idx]
        logging.info(f"Tool: {tool_id}, Similarity: {similarity}")
    
    # Step 6: Create a subset of tools to include in the GPT prompt
    subset_tools = [tool for tool in tools if tool['name'] in top_tool_ids]
    
    # Step 7: Construct the GPT prompt with the subset of tools
    prompt = f"""
Based on the current state and goal, please suggest possible next tools to apply. The response must be in the following JSON format:

{{
    "current_conditions": {json.dumps(list(state.conditions))},
    "current_conclusions": {json.dumps(list(state.conclusions))},
    "goal_conclusions": {json.dumps(list(goal_conclusions))},
    "next_possible_tools": ["Tool1", "Tool2", ...]
}}

Use the following tool descriptions to determine applicable tools:

"""
    for tool in subset_tools:
        prompt += f"Tool Name: {tool['name']}\nDescription: {tool['description']}\nConditions: {', '.join(tool['conditions'])}\nConclusions: {', '.join(tool['conclusions'])}\n\n"
    
    # Step 8: Add the prompt as a user message to the state's messages
    user_message = {
        "role": "user",
        "content": prompt.strip()
    }
    state.messages.append(user_message)
    logging.info(f"Added user message to state: {user_message}")
    
    # Step 9: Call GPT to get the next possible tools
    response = gpt_generate_use_azure(prompt)
    logging.info(f"GPT Response for Possible Actions: {response}")
    
    # Step 10: Extract JSON from the response
    data = extract_json(response)
    actions = data.get('next_possible_tools', [])
    
    # Step 11: Filter actions to ensure they are valid tool names
    valid_actions = [action for action in actions if action in top_tool_ids]
    logging.info(f"Next possible tools after filtering: {valid_actions}")
    return valid_actions



def apply_action(state: State, action_name: str) -> State:
    """
    Applies a tool to the current state following Steps 3 to 7:
    Step 3: Get arguments for the tool by calling GPT.
    Step 4: Add a function call message.
    Step 5: Execute the tool's function (now includes requirements).
    Step 6: Add an observation message.
    Step 7: Generate and add the assistant's reply by calling GPT.
    """
    # Find the tool by its name
    tool = next((tool for tool in tools if tool['name'] == action_name), None)
    if tool is None:
        logging.error(f"Tool '{action_name}' does not exist.")
        return state  # Tool does not exist, state remains unchanged

    # Step 3: Get arguments for the tool by calling GPT
    prompt = f"""
Based on the current conversation and the tool '{action_name}', please generate the required arguments to call the tool.
Provide the arguments in JSON format.

Conversation Messages:
{json.dumps(state.messages, indent=2)}

Tool Description:
Name: {tool['name']}
Description: {tool['description']}
Parameters: {json.dumps(tool.get('parameters', {}), indent=2)}

Please provide your answer in the following JSON format:
{{
    "arguments": {{}}
}}
Only provide the JSON output.
"""

    args_response = gpt_generate_use_azure(prompt)
    logging.info(f"GPT Response for Arguments: {args_response}")
    args_data = extract_json(args_response)
    arguments = args_data.get('arguments', {})

    if not arguments:
        logging.error(f"No arguments provided for tool '{action_name}'.")
        return state  # No arguments provided, state remains unchanged

    # Step 4: Add a function call message
    function_message = {
        "role": "function",
        "content": json.dumps({"name": tool['name'], "arguments": arguments}, ensure_ascii=False)
    }

    # Step 5: Execute the tool's function (includes requirements)
    try:
        # Pass the tool's requirements to the execute_tool_function
        requirements = tool.get('requirement', [])
        logging.info(f"Requirements for tool '{action_name}': {requirements}")
        result = execute_tool_function(requirements, tool['function_body'], arguments)
        
        # Serialize the result if it's a NumPy array
        if isinstance(result, np.ndarray):
            result_serializable = result.tolist()
        else:
            result_serializable = result
    except Exception as e:
        logging.error(f"Error executing tool '{action_name}': {e}")
        return state  # Execution failed, state remains unchanged

    # Step 6: Add an observation message
    observation_message = {
        "role": "observation",
        "content": json.dumps(result_serializable, ensure_ascii=False)
    }

    # Step 7: Generate and add the assistant's reply by calling GPT
    assistant_prompt = f"""
Based on the following conversation (including function calls and observations), use a natural language sentence to reply.

Conversation Messages:
{json.dumps(state.messages + [function_message, observation_message], indent=2)}
"""
    assistant_response = gpt_generate_use_azure(assistant_prompt)
    logging.info(f"GPT Response for Assistant: {assistant_response}")

    assistant_message = {
        "role": "assistant",
        "content": assistant_response.strip()
    }

    # Update messages
    new_messages = state.messages.copy()
    new_messages.extend([function_message, observation_message, assistant_message])

    # Step 8: Generate new conditions and conclusions based on tool's descriptions and function output
    conditions_prompt = f"""
Based on the tool's conditions and conclusions descriptions, the arguments used, and the function output, please determine the new conditions and conclusions to update the state. Include the actual argument values and function outputs in the new conditions and conclusions.

Tool Conditions:
{json.dumps(tool['conditions'], indent=2)}

Tool Conclusions:
{json.dumps(tool['conclusions'], indent=2)}

Arguments Used:
{json.dumps(arguments, indent=2)}

Function Output:
{json.dumps(result_serializable, indent=2)}

Please provide your answer in the following JSON format:
{{
    "new_conditions": [...],
    "new_conclusions": [...]
}}
Only provide the JSON output.
"""

    conditions_response = gpt_generate_use_azure(conditions_prompt)
    logging.info(f"GPT Response for Conditions and Conclusions: {conditions_response}")
    conditions_data = extract_json(conditions_response)
    new_conditions = conditions_data.get('new_conditions', [])
    new_conclusions = conditions_data.get('new_conclusions', [])

    # Update state with new conditions and conclusions
    updated_conditions = state.conditions.union(new_conditions)
    updated_conclusions = state.conclusions.union(new_conclusions)

    # Create a new state with updated messages, conditions, and conclusions
    new_state = State(
        conditions=list(updated_conditions),
        conclusions=list(updated_conclusions),
        messages=new_messages
    )

    logging.info(f"Applied tool '{action_name}' successfully. Updated state: {new_state}")

    return new_state


def select(node: Node) -> Node:
    """
    Selects the best child node using the UCB1 formula.
    """
    while node.children:
        node = select_best_child(node)
    return node

def select_best_child(node: Node, c_param: float = 1.41) -> Node:
    """
    Selects the child with the highest UCB1 score.
    """
    choices_weights = []
    for child in node.children:
        if child.visits == 0:
            # Prevent division by zero, give high score to unvisited nodes
            score = float('inf')
        else:
            exploitation = child.reward / child.visits
            exploration = c_param * math.sqrt((2 * math.log(node.visits) / child.visits))
            score = exploitation + exploration
        choices_weights.append(score)
    
    max_weight = max(choices_weights)
    best_child = node.children[choices_weights.index(max_weight)]
    return best_child

def simulate(node: Node, goal_conclusions: set, max_depth: int = 5) -> int:
    """
    Simulates a random playout from the given node to a maximum depth.
    Returns a reward of 1 if the goal is achieved, else 0.
    Logs the simulation process.
    """
    current_state = node.state
    depth = 0
    while depth < max_depth:
        if current_state.is_goal_state_gpt(goal_conclusions):
            logging.info("Goal conclusions achieved during simulation.")
            return 1  # Success reward
        actions = get_possible_actions(current_state, goal_conclusions)
        if not actions:
            logging.info("No available tools during simulation. Ending simulation.")
            break
        action = random.choice(actions)
        logging.info(f"Simulation: selected tool '{action}'.")
        new_state = apply_action(current_state, action)
        # If state does not change, stop simulation
        if new_state == current_state:
            logging.info(f"Simulation: applying tool '{action}' did not change state. Ending simulation.")
            break
        current_state = new_state
        depth += 1
    return 0  # Failure penalty

def backpropagate(node: Node, reward: int):
    """
    Propagates the reward up the tree.
    """
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent

def find_success_nodes(node: Node, goal_conclusions: set, success_nodes: List[Node]):
    """
    Recursively traverses the tree to find all nodes that satisfy the goal.

    Parameters:
    - node (Node): The current node to check.
    - goal_conclusions (set): The set of goal conclusions.
    - success_nodes (list): The list to append successful nodes to.
    """
    if node.state.is_goal_state_gpt(goal_conclusions):
        success_nodes.append(node)
    for child in node.children:
        find_success_nodes(child, goal_conclusions, success_nodes)

def mcts(root: Node, goal_conclusions: set, iterations: int = 50) -> Node:
    """
    Performs MCTS starting from the root node.
    Logs each iteration's progress.

    Parameters:
    - root (Node): The root node of the MCTS tree.
    - goal_conclusions (set): The set of goal conclusions.
    - iterations (int): Number of iterations to perform.

    Returns:
    - Node: The best node found that satisfies the goal, or None if not found.
    """
    for i in range(1, iterations + 1):
        logging.info(f"MCTS Iteration {i}/{iterations}")

        # Selection
        node = select(root)
        logging.info(f"Selected node with state: {node.state}")

        # If the node already satisfies the goal, assign a positive reward and skip expansion
        if node.state.is_goal_state_gpt(goal_conclusions):
            logging.info("Selected node already satisfies the goal. Assigning reward.")
            backpropagate(node, 1)
            logging.info(f"Iteration {i} completed with reward: 1")
            continue

        # Expansion
        actions = get_possible_actions(node.state, goal_conclusions)
        for action in actions:
            new_state = apply_action(node.state, action)
            if new_state != node.state:
                child_node = Node(state=new_state, parent=node, action=action)
                node.children.append(child_node)
                logging.info(f"Expanded node with tool '{action}'. New state: {new_state}")

        # If no actions are possible after expansion, assign a penalty
        if not node.children:
            logging.info("No actions available for the selected node. Assigning penalty.")
            backpropagate(node, 0)
            logging.info(f"Iteration {i} completed with reward: 0")
            continue

        # Select a child node to simulate
        node_to_simulate = random.choice(node.children)
        logging.info(f"Simulating node with state: {node_to_simulate.state}")

        # Simulation
        reward = simulate(node_to_simulate, goal_conclusions)

        # Backpropagation
        backpropagate(node_to_simulate, reward)
        logging.info(f"Iteration {i} completed with reward: {reward}")

    # After all iterations, traverse the tree to find all nodes that satisfy the goal
    success_nodes = []
    find_success_nodes(root, goal_conclusions, success_nodes)

    if success_nodes:
        # Select the node with the highest visits
        best_node = max(success_nodes, key=lambda n: n.visits, default=None)
        logging.info(f"Best node selected: Applied tool '{best_node.action}', Visits: {best_node.visits}, Total Reward: {best_node.reward}")
    else:
        # If no successful nodes found, select the child with the most visits
        best_node = max(root.children, key=lambda n: n.visits, default=None)
        if best_node and best_node.state.is_goal_state_gpt(goal_conclusions):
            logging.info(f"Best child selected: Applied tool '{best_node.action}', Visits: {best_node.visits}, Total Reward: {best_node.reward}")
        else:
            logging.info("No best child found that satisfies the goal.")

    return best_node

def extract_solution(node: Node) -> List[Dict[str, Any]]:
    """
    Extracts the sequence of actions from the root to the given node.

    Returns:
    - List of dictionaries containing 'action' and 'state'.
    """
    solution = []
    while node.parent is not None:
        solution.append({"action": node.action, "state": node.state})
        node = node.parent
    solution.reverse()
    return solution

# ----------------------------
# 6. Main Execution Block
# ----------------------------

if __name__ == "__main__":
    # Example Initial Question
    initial_question = "Now I have two vectors [1, 2, 3] and [4, 5, 6], I want to get the sum of these two vectors."

    # Step 1: Parse the initial question into JSON format
    parsed_initial = parse_initial_question(initial_question)
    print("parsed_initial_question:", parsed_initial)
    initial_conditions = parsed_initial.get("initial_conditions", [])
    initial_conclusions = parsed_initial.get("initial_conclusions", [])
    goal_conclusions = set(parsed_initial.get("goal_conclusions", {}))

    # Initialize the state with parsed conditions, conclusions, and initial message
    initial_state = State(
        conditions=initial_conditions,
        conclusions=initial_conclusions,
        messages=[
            {
                "role": "user",
                "content": initial_question
            },
            {
                "role": "assistant",
                "content": f"The parsed initial question is: {parsed_initial}"
            }
        ]
    )

    # Initialize the root node
    root_node = Node(state=initial_state)

    # Print initial problem details
    print(f"Starting MCTS for the provided initial question.")
    print(f"Initial Conditions: {initial_state.conditions}")
    print(f"Goal Conclusions: {goal_conclusions}\n")
    # Log initial problem details
    logging.info(f"Starting MCTS for the provided initial question.")
    logging.info(f"Initial Conditions: {initial_state.conditions}")
    logging.info(f"Goal Conclusions: {goal_conclusions}")

    # Run MCTS to find the best node
    best_node = mcts(root_node, goal_conclusions, iterations=10)

    # Check if a solution was found
    if best_node and best_node.state.is_goal_state_gpt(goal_conclusions):
        solution = extract_solution(best_node)
        # Print the solution
        print("\nSolution Found! Tool Application Sequence:")
        # Log the solution
        logging.info("Solution Found! Tool Application Sequence:")
        for step in solution:
            action = step['action']
            state = step['state']
            print(f"Applied Tool: {action}, New State: {state}")
            logging.info(f"Applied Tool: {action}, New State: {state}")
        # Optionally, print the final messages
        print("\nFinal Conversation Messages:")
        for message in best_node.state.messages:
            print(f"{message['role']}: {message['content']}")
            logging.info(f"{message['role']}: {message['content']}")
    else:
        print("No solution found.")
        logging.info("No solution found.")

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
    logging.info("End of MCTS execution.")
    logging.info("--------------------------------------------------")
