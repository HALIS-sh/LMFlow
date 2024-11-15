MESSAGE_PROMPT = '''
## Task Description

You will generate a structured conversation that represents a sequence of function calls based on a given mathematical problem, its solution, and the corresponding Python code. The conversation should illustrate how the problem is solved step-by-step, including any necessary function calls.

### Input Format:

1. **[problem]:** A brief description of a mathematical problem.
2. **[solution]:** A detailed solution to the problem, including any intermediate steps.
3. **[python solution]:** The Python code used to calculate the solution, which may include custom functions.

### Output Format:

The output should be in JSON format, depicting a dialogue between a user and an assistant. The dialogue should include observations and feedback after each function call.
If the provided Python solution is missing any necessary functions, include those functions in the output. Each missing function should be presented in standard Python format, with a brief description of its purpose.
The functions to include are those for basic operations such as addition (Add), multiplication (multiply), exponentiation (power), modulus (mod), division (divide) and so on. Do not include them if they are not used.


### Example Input:

{example}

### Input:

[problem]:
{problem}

[solution]:
{solution}

[python solution]:
{python_solution}
'''

EXAMPLE = '''```
[problem]:  
If \( x = 2 \) and \( y = 5 \), then what is the value of \( \frac{x^4 + 2y^2}{6} \)?

[solution]:  
We have  
\[ 
\frac{x^4 + 2y^2}{6} = \frac{2^4 + 2(5^2)}{6} = \frac{16 + 50}{6} = \frac{66}{6} = \boxed{11}. 
\]

[python solution]:    
def calculate_power(base, exponent):
    # Function to calculate the power of a number.
    return base ** exponent

def divide_by(value, divisor):
    # Function to divide a value by a divisor.
    return value / divisor

# Example code execution
if __name__ == "__main__":
    x = 2
    y = 5
    expression_value = calculate_power(x, 4) + 2 * calculate_power(y, 2)
    answer = divide_by(expression_value, 6)
    print(answer)
```

### Example Output:

Dialogue Messages:

```json
[
    {
        "role": "user",
        "content": "If \( x = 2 \) and \( y = 5 \), then what is the value of \( \frac{x^4 + 2y^2}{6} \)?"
    },
    {
        "role": "function",
        "content": {"name": "calculate_power", "arguments": {"base": 2, "exponent": 4}}
    },
    {
        "role": "observation",
        "content": {"power": 16}
    },
    {
        "role": "assistant",
        "content": "The power of x is 16."
    },
    {
        "role": "user",
        "content": "Now, with \( x^4 = 16 \), what is the next step for \( \frac{x^4 + 2y^2}{6} \)?" 
    },
    {
        "role": "function",
        "content": {"name": "calculate_power", "arguments": {"base": 5, "exponent": 2}}
    },
    {
        "role": "observation",
        "content": {"Square": 25}
    },
    {
        "role": "assistant",
        "content": "The square of y is 25."
    },
    {
        "role": "user",
        "content": "Given \( y^2 = 25 \), what is \( 2y^2 \)?"
    },
    {
        "role": "function",
        "content": {"name": "multiply", "arguments": {"value1": 2, "value2": 25}}
    },
    {
        "role": "observation",
        "content": {"multiple": 50}
    },
    {
        "role": "assistant",
        "content": "The value of \( 2y^2 \) is 50."
    },
    {
        "role": "user",
        "content": "Now, what is the total of \( x^4 + 2y^2 \)?"
    },
    {
        "role": "function",
        "content": {"name": "Add", "arguments": {"value1": 16, "value2": 50}}
    },
    {
        "role": "observation",
        "content": {"add_result": 66}
    },
    {
        "role": "assistant",
        "content": "The sum \( x^4 + 2y^2 \) is 66."
    },
    {
        "role": "user",
        "content": "Finally, what is \( \frac{66}{6} \)?"
    },
    {
        "role": "function",
        "content": {"name": "divide_by", "arguments": {"value": 66, "divisor": 6}}
    },
    {
        "role": "observation",
        "content": {"divide_result": 11}
    },
    {
        "role": "assistant",
        "content": "The final result is \( 11 \)."
    }
]
```

Missing Python Functions:

```python
def Add(value1, value2):
    """
    Add two values.

    Args:
        value1 (int or float): The first value.
        value2 (int or float): The second value.

    Returns:
        int or float: The sum of value1 and value2.
    """
    return value1 + value2

def multiply(value1, value2):
    """
    Multiply two values.

    Args:
        value1 (int or float): The first value.
        value2 (int or float): The second value.

    Returns:
        int or float: The product of value1 and value2.
    """
    return value1 * value2
```
'''