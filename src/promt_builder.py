from src.parser import FunctionDefinition


def build_prompt(
    user_prompt: str, function_definitions: list[FunctionDefinition]
) -> str:
    prompt = "Available functions:\n"
    for fn in function_definitions:
        prompt += '- ' + fn.name
        prompt += "("
        i = 0
        for _, value in fn.parameters.items():
            prompt += value.type
            if i != len(fn.parameters) - 1:
                prompt += ', '
            i += 1
        prompt += ')'
        prompt += ": " + fn.description
        prompt += ". Returns: " + fn.returns.type + '.\n\n'
    prompt += "User request: " + user_prompt + "\n"
    prompt += "Select the single best function for the user request.\n"
    return prompt


def function_names(
    function_definitions: list[FunctionDefinition],
) -> list[str]:
    Fn_names = []
    for fn in function_definitions:
        Fn_names.append(fn.name + '"')
    return Fn_names
