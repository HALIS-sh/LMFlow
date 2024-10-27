import json
import math
import random
import logging
import re
from openai import OpenAI
import httpx


# Configure logging
logging.basicConfig(
    filename='mcts_gpt_integration.log',  # Log file name
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

# Global counters to track API usage
api_call_count = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

def gpt_generate(prompt, model='gpt-4', max_tokens=150):
    """
    Calls the GPT-4 API to generate a response and logs relevant information.
    """
    global api_call_count, total_prompt_tokens, total_completion_tokens, total_tokens

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides responses in the specified JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return ""

    # Update global counters
    api_call_count += 1
    # print('completion', completion)
    # print('type of completion', type(completion))
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

    message = completion.choices[0].message.content.strip()
    return message

class State:
    """
    Represents the current state, including conditions and conclusions.
    After applying a tool, the tool's conclusions are appended to the conditions to influence subsequent decisions.
    """
    def __init__(self, conditions, conclusions):
        self.conditions = set(conditions)
        self.conclusions = set(conclusions)

    def is_goal_state(self, goal_conclusions):
        return goal_conclusions.issubset(self.conclusions)

    def __eq__(self, other):
        return self.conditions == other.conditions and self.conclusions == other.conclusions

    def __hash__(self):
        return hash((frozenset(self.conditions), frozenset(self.conclusions)))

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

# Define the list of tools
tools = [
    {
        "name": "Medical Diagnosis Tool",
        "description": "A tool for diagnosing diseases based on symptoms.",
        "conditions": ["HasSymptoms"],
        "conclusions": ["PossibleDiseases"]
    },
    {
        "name": "Flu Test Tool",
        "description": "A tool for testing the presence of flu virus.",
        "conditions": ["PossibleDiseases"],
        "conclusions": ["DiagnosedWithFlu"]
    },
    {
        "name": "Data Preprocessing Tool",
        "description": "A tool for cleaning and preprocessing datasets.",
        "conditions": ["DataSetAvailable"],
        "conclusions": ["CleanData"]
    },
    {
        "name": "Model Training Tool",
        "description": "A tool for training machine learning models.",
        "conditions": ["CleanData"],
        "conclusions": ["ModelTrained"]
    },
    {
        "name": "Question Understanding Tool",
        "description": "A tool for analyzing user questions.",
        "conditions": ["UserQuestion"],
        "conclusions": ["QuestionAnalyzed"]
    },
    {
        "name": "Information Retrieval Tool",
        "description": "A tool for retrieving information from a knowledge base.",
        "conditions": ["QuestionAnalyzed"],
        "conclusions": ["RelevantInformation"]
    },
    {
        "name": "Answer Generation Tool",
        "description": "A tool for generating answers.",
        "conditions": ["RelevantInformation"],
        "conclusions": ["AnswerProvided"]
    },
    # Add more tools as needed
]

def get_possible_actions(state, goal_conclusions):
    """
    Calls GPT-4 to get the next possible tools based on the current state and goal conclusions.
    GPT-4 is forced to output a JSON containing current conditions, conclusions, and next possible tools.
    """
    prompt = f"""
Based on the current state and goal, please generate the next possible tools to apply. The response must be in the following JSON format:

{{
    "current_conditions": {json.dumps(list(state.conditions))},
    "current_conclusions": {json.dumps(list(state.conclusions))},
    "goal_conclusions": {json.dumps(list(goal_conclusions))},
    "next_possible_tools": ["Tool1", "Tool2", ...]
}}

Use the following tool descriptions to determine applicable tools:

"""
    for tool in tools:
        prompt += f"Tool Name: {tool['name']}\nConditions: {', '.join(tool['conditions'])}\nConclusions: {', '.join(tool['conclusions'])}\nDescription: {tool['description']}\n\n"

    response = gpt_generate(prompt)

    # Extract JSON from the response
    data = extract_json(response)
    actions = data.get('next_possible_tools', [])

    # Filter actions to ensure they are valid tool names
    valid_actions = [action for action in actions if action in [tool['name'] for tool in tools]]
    logging.info(f"Next possible tools: {valid_actions}")
    return valid_actions


def apply_action(state, action_name):
    """
    Applies a tool to the current state by updating the state's conclusions to the tool's conclusions
    and appending the tool's conclusions to the conditions.
    Logs relevant information.
    
    Parameters:
    - state (State): The current state containing conditions and conclusions.
    - action_name (str): The name of the tool to apply.
    
    Returns:
    - State: The new state after applying the tool. If the tool cannot be applied, returns the original state.
    """
    # Find the tool by its name
    tool = next((tool for tool in tools if tool['name'] == action_name), None)
    if tool is None:
        logging.error(f"Tool '{action_name}' does not exist.")
        return state  # Tool does not exist, state remains unchanged

    # Check if the tool's conditions are met
    if not set(tool['conditions']).issubset(state.conditions.union(state.conclusions)):
        logging.info(f"Tool '{action_name}' conditions are not met and cannot be applied.")
        return state  # Conditions not met, state remains unchanged

    # Apply the tool:
    # - Append tool's conclusions to conditions
    # - Set conclusions to tool's conclusions
    new_conditions = state.conditions.copy()
    new_conditions.update(tool['conclusions'])  # Append conclusions to conditions

    new_conclusions = set(tool['conclusions'])  # Directly update conclusions to tool's conclusions

    new_state = State(new_conditions, new_conclusions)

    # Check if the state has changed
    if new_state != state:
        logging.info(f"Tool '{action_name}' applied successfully. Updated state: {new_state}.")
    else:
        logging.info(f"Applying tool '{action_name}' did not change the state.")

    return new_state


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

def expand(node, goal_conclusions):
    """
    Expands the node by generating its children based on possible actions.
    Logs the expansion details.
    """
    actions = get_possible_actions(node.state, goal_conclusions)
    for action in actions:
        new_state = apply_action(node.state, action)
        logging.info(f"Expanding node with tool '{action}'. New state: {new_state}")
        # Only add child if state has changed
        if new_state != node.state:
            child_node = Node(state=new_state, parent=node, action=action)
            node.children.append(child_node)
            logging.info(f"Expanded node with tool '{action}'. New state: {new_state}")

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

        # If the simulation leads to the goal, add to success_nodes
        if node_to_simulate.state.is_goal_state(goal_conclusions):
            logging.info(f"Simulation achieved the goal with tool '{node_to_simulate.action}'.")
            # Optionally, keep track of success nodes here if needed

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

def extract_json(response):
    """
    Extracts JSON block from the GPT response.
    Assumes JSON is enclosed within triple backticks.
    """
    try:
        # Use regex to find content within triple backticks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            return data
        else:
            # Fallback: find the first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                data = json.loads(json_str)
                return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
    return {}


# Sample initial problems
initial_problems = [
    {
        "problem_id": 1,
        "initial_conditions": ["HasSymptoms"],
        "initial_conclusions": [],
        "goal_conclusions": ["DiagnosedWithFlu"]
    },
    {
        "problem_id": 2,
        "initial_conditions": ["DataSetAvailable"],
        "initial_conclusions": [],
        "goal_conclusions": ["ModelTrained"]
    },
    {
        "problem_id": 3,
        "initial_conditions": ["UserQuestion"],
        "initial_conclusions": [],
        "goal_conclusions": ["AnswerProvided"]
    },
    # Add more problems as needed
]

if __name__ == "__main__":
    import sys
    for problem in initial_problems:
        # # Select an initial problem (change the index to select a different problem)
        # problem_index = 2  # Change to 1 or 2 for different problems
        # if problem_index >= len(initial_problems):
        #     print("Invalid problem index.")
        #     sys.exit(1)

        # problem = initial_problems[problem_index]
        initial_state = State(conditions=problem['initial_conditions'], conclusions=problem['initial_conclusions'])
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
