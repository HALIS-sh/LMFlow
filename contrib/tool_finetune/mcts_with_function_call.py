import json
import math
import random
import logging
import re
import os
import requests
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    filename='mcts_gpt_integration_with_tools.log',  # Log file name
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

# Configuration
API_KEY = os.environ.get("AZUREAI_API_KEY")
ENDPOINT = os.environ.get("AZUREAI_ENDPOINT_URL")
ENDPOINT = "https://your-endpoint.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Global counters to track API usage
api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

def gpt_generate_use_azure(prompt, model='gpt-4', max_tokens=150):
    """
    Calls the GPT-4 API to generate a response and logs relevant information.
    """
    global api_call_count, total_prompt_tokens, total_completion_tokens, total_tokens

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
    except requests.RequestException as e:
        logging.error(f"Failed to make the request. Error: {e}")
        return ""
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response. Error: {e}")
        return ""

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

class State:
    """
    Represents the current state, including conditions, conclusions, and messages.
    """
    def __init__(self, conditions, conclusions, messages):
        self.conditions = set(conditions)
        self.conclusions = set(conclusions)
        self.messages = messages  # List of dialog messages

    def is_goal_state(self, goal_conclusions):
        """
        Determines whether the goal has been achieved based on the state's conclusions.
        """
        return goal_conclusions.issubset(self.conclusions)

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
    def __init__(self, state, parent=None, action=None):
        self.state = state         # Current state
        self.parent = parent       # Parent node
        self.action = action       # Action (tool application) from parent to current node
        self.children = []         # List of child nodes
        self.visits = 0            # Number of times node has been visited
        self.reward = 0            # Accumulated reward

# Load tools from JSON or define them directly
tools = [
    {
        "name": "add_vectors",
        "description": "Add two 3-dimensional vectors.",
        "conditions": [
            "vector_a (list of numbers): First vector with 3 components (x, y, z).",
            "vector_b (list of numbers): Second vector with 3 components (x, y, z)."
        ],
        "conclusions": [
            "The result of adding vector_a and vector_b is obtained."
        ],
        "requirement": [
            "import numpy as np"
        ],
        "function_body": """
def add_vectors(vector_a, vector_b):
    import numpy as np
    return np.array(vector_a) + np.array(vector_b)
"""
    },
    {
        "name": "get_news_headlines",
        "description": "Get the latest news headlines",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "The country for which to fetch news"}
            },
            "required": ["country"]
        },
        "conditions": [
            "Country name is provided."
        ],
        "conclusions": [
            "Latest news headlines for the specified country are obtained."
        ],
        "function_body": """
def get_news_headlines(country):
    headlines = [
        f"Breaking news in {country}: Headline 1",
        f"Latest update in {country}: Headline 2",
        f"Top story in {country}: Headline 3",
        f"News flash in {country}: Headline 4"
    ]
    return {"headlines": headlines}
"""
    },
    # Add other tools as needed
]

# Helper function to execute a tool's function_body
def execute_tool_function(function_body: str, args: Dict[str, Any]) -> Any:
    """
    Executes the provided function_body as Python code with the given arguments.
    Args:
        function_body (str): The Python function code.
        args (dict): The arguments to pass to the function.
    Returns:
        Any: The result of the function execution.
    """
    local_vars = {}
    exec(function_body, {}, local_vars)  # Execute the function body
    func_name_match = re.search(r"def (\w+)\(", function_body)
    if func_name_match:
        func_name = func_name_match.group(1)
    else:
        raise ValueError("Function name could not be determined from function_body.")
    if func_name in local_vars:
        return local_vars[func_name](**args)  # Call the extracted function
    else:
        raise ValueError(f"Function '{func_name}' not found after execution.")

def extract_json(response):
    """
    Extracts JSON block from the GPT response.
    Assumes JSON is enclosed within curly braces.
    """
    try:
        # Use regex to find the JSON object
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
    return {}

def get_possible_actions(state, goal_conclusions):
    """
    Calls GPT-4 to get the next possible tools based on the current state and goal conclusions.
    """
    prompt = f"""
Based on the following conversation, current conditions, and goal, please suggest possible next tools to apply.

Conversation Messages:
{json.dumps(state.messages, indent=2)}

Current Conditions:
{json.dumps(list(state.conditions), indent=2)}

Goal Conclusions:
{json.dumps(list(goal_conclusions), indent=2)}

Available Tools:
"""
    for tool in tools:
        prompt += f"Tool Name: {tool['name']}\nDescription: {tool['description']}\nConditions: {json.dumps(tool.get('conditions', []), indent=2)}\nConclusions: {json.dumps(tool.get('conclusions', []), indent=2)}\n\n"

    prompt += """
Please provide your answer in the following JSON format:
{
    "next_possible_tools": ["Tool1", "Tool2", ...]
}
Only provide the JSON output.
"""

    response = gpt_generate_use_azure(prompt)
    data = extract_json(response)
    actions = data.get('next_possible_tools', [])

    # Filter actions to ensure they are valid tool names
    valid_actions = [action for action in actions if action in [tool['name'] for tool in tools]]
    logging.info(f"Next possible tools: {valid_actions}")
    return valid_actions

def apply_action(state, action_name):
    """
    Applies a tool to the current state by executing its function_body.
    Updates the conditions and conclusions based on GPT's interpretation of the tool's descriptions, arguments, and function output.
    """
    # Find the tool by its name
    tool = next((tool for tool in tools if tool['name'] == action_name), None)
    if tool is None:
        logging.error(f"Tool '{action_name}' does not exist.")
        return state  # Tool does not exist, state remains unchanged

    # Use GPT to get the tool's arguments
    prompt = f"""
Based on the following conversation and the tool '{action_name}', please generate the required arguments to call the tool.
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

    response = gpt_generate_use_azure(prompt)
    data = extract_json(response)
    arguments = data.get('arguments', {})

    # Execute the tool's function_body
    try:
        result = execute_tool_function(tool['function_body'], arguments)

        # Prepare the function output for JSON serialization
        if isinstance(result, (np.ndarray, list)):
            result_serializable = result.tolist() if hasattr(result, 'tolist') else result
        elif isinstance(result, dict):
            result_serializable = result
        else:
            result_serializable = result

        # Use GPT to generate new conditions and conclusions, including actual arguments and outputs
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
        conditions_data = extract_json(conditions_response)
        new_conditions = conditions_data.get('new_conditions', [])
        new_conclusions = conditions_data.get('new_conclusions', [])

        # Update messages
        new_messages = state.messages.copy()
        # Add function call message
        new_messages.append({
            "role": "function",
            "content": json.dumps({"name": tool['name'], "arguments": arguments}, ensure_ascii=False)
        })
        # Add observation message
        new_messages.append({
            "role": "observation",
            "content": json.dumps(result_serializable, ensure_ascii=False)
        })
        # Use GPT to generate assistant's reply
        assistant_prompt = f"""
Based on the following conversation (including function calls and observations), generate the assistant's reply.

Conversation Messages:
{json.dumps(new_messages, indent=2)}
"""
        assistant_response = gpt_generate_use_azure(assistant_prompt)
        new_messages.append({
            "role": "assistant",
            "content": assistant_response.strip()
        })

        # Update state with new conditions and conclusions
        updated_conditions = state.conditions.union(new_conditions)
        updated_conclusions = state.conclusions.union(new_conclusions)

        # Create a new state
        new_state = State(
            conditions=updated_conditions,
            conclusions=updated_conclusions,
            messages=new_messages
        )
        logging.info(f"Applied tool '{action_name}' successfully. Updated state.")
        return new_state

    except Exception as e:
        logging.error(f"Error executing tool '{action_name}': {e}")
        return state

def select(node):
    """
    Selects the best child node using the UCB1 formula.
    """
    while node.children:
        node = select_best_child(node)
    return node

def select_best_child(node, c_param=1.41):
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

def simulate(node, goal_conclusions, max_depth=5):
    """
    Simulates a random playout from the given node to a maximum depth.
    Returns a reward of 1 if the goal is achieved, else 0.
    Logs the simulation process.
    """
    current_state = node.state
    depth = 0
    while depth < max_depth:
        if current_state.is_goal_state(goal_conclusions):
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

def backpropagate(node, reward):
    """
    Propagates the reward up the tree.
    """
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent

def find_success_nodes(node, goal_conclusions, success_nodes):
    """
    Recursively traverses the tree to find all nodes that satisfy the goal.

    Parameters:
    - node (Node): The current node to check.
    - goal_conclusions (set): The set of goal conclusions.
    - success_nodes (list): The list to append successful nodes to.
    """
    if node.state.is_goal_state(goal_conclusions):
        success_nodes.append(node)
    for child in node.children:
        find_success_nodes(child, goal_conclusions, success_nodes)

def mcts(root, goal_conclusions, iterations=50):
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
        if node.state.is_goal_state(goal_conclusions):
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
        if best_node and best_node.state.is_goal_state(goal_conclusions):
            logging.info(f"Best child selected: Applied tool '{best_node.action}', Visits: {best_node.visits}, Total Reward: {best_node.reward}")
        else:
            logging.info("No best child found that satisfies the goal.")

    return best_node

def extract_solution(node):
    """
    Extracts the sequence of actions from the root to the given node.
    """
    solution = []
    while node.parent is not None:
        solution.append((node.action, node.state))
        node = node.parent
    solution.reverse()
    return solution

# Sample initial problems
initial_problems = [
    {
        "problem_id": 1,
        "initial_conditions": [],
        "initial_conclusions": [],
        "initial_messages": [
            {
                "role": "user",
                "content": "Can you tell me the latest news headlines for the United States?"
            }
        ],
        "goal_conclusions": {"Latest news headlines for the specified country are obtained."}
    },
    {
        "problem_id": 2,
        "initial_conditions": [],
        "initial_conclusions": [],
        "initial_messages": [
            {
                "role": "user",
                "content": "Please add the vectors [1, 2, 3] and [4, 5, 6]."
            }
        ],
        "goal_conclusions": {"Resulting vector [5, 7, 9] obtained as np.ndarray."}
    },
    # Add more problems as needed
]

if __name__ == "__main__":
    for problem in initial_problems:
        initial_state = State(
            conditions=problem['initial_conditions'],
            conclusions=problem['initial_conclusions'],
            messages=problem['initial_messages']
        )
        goal_conclusions = set(problem['goal_conclusions'])
        root_node = Node(state=initial_state)

        # Print initial problem details
        print(f"Starting MCTS for Problem ID: {problem['problem_id']}")
        print(f"Initial Conditions: {initial_state.conditions}")
        print(f"Goal Conclusions: {goal_conclusions}\n")
        # Log initial problem details
        logging.info(f"Starting MCTS for Problem ID: {problem['problem_id']}")
        logging.info(f"Initial Conditions: {initial_state.conditions}")
        logging.info(f"Goal Conclusions: {goal_conclusions}")

        best_node = mcts(root_node, goal_conclusions, iterations=10)

        if best_node and best_node.state.is_goal_state(goal_conclusions):
            solution = extract_solution(best_node)
            # Print the solution
            print("\nSolution Found! Tool Application Sequence:")
            # Log the solution
            logging.info("Solution Found! Tool Application Sequence:")
            for action, state in solution:
                print(f"Applied Tool: {action}, New State: {state}")
                logging.info(f"Applied Tool: {action}, New State: {state}")
            # Output the generated conversation messages
            print("\nGenerated Conversation Messages:")
            for message in best_node.state.messages:
                print(f"{message['role']}: {message['content']}")
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
