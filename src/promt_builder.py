from src.parser import FunctionDefinition

def build_prompt(user_prompt: str, function_definitions: list[FunctionDefinition]) -> str:
    prompt = "You are a function calling assistant. Given the user's request, output a JSON object with the function name and arguments.\n\n"
    prompt += "Available functions:\n"
    for fn in function_definitions:
        prompt += '- ' + fn.name
        prompt += "("
        for key , value in fn.parameters.items():
            prompt += f"{value.type}" + ', '
        prompt = prompt[:-2]
        prompt += ')'
        prompt += ": "  + fn.description
        prompt += ". Returns: " + fn.returns.type + '.\n\n'
    prompt += "-fn_unknown: unknow function for that promt"
    prompt += "\n"
    prompt += "User request: " + user_prompt + "\n"
    prompt += "\n\nIMPORTANT: Choose the function"
    "whose description BEST matches the user prompt."
    prompt += "\nRead each function description carefully before choosing."
    prompt += "\nRespond with only the function name that matches.\n"
    prompt += 'Output: {"name":'
    return prompt

def function_names(function_definitions: list[FunctionDefinition]):
    Fn_names = []
    for fn in function_definitions:
        Fn_names.append(fn.name)
    return Fn_names

# You are a function calling assistant. Given the user's request, output a JSON object with the function name and arguments.

# Available functions:
# - fn_add_numbers: Add two numbers together and return their sum. Parameters: a (number), b (number). Returns: number.
# - fn_greet: Generate a greeting message for a person by name. Parameters: name (string). Returns: string.
# - fn_reverse_string: Reverse a string and return the reversed result. Parameters: s (string). Returns: string.

# User request: Reverse the string 'hello'


