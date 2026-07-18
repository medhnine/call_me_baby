import argparse

from .promt_builder import build_prompt
from .parser import load_function_definitions, load_promts


def function_caller(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    res = load_function_definitions(args.functions_definition)
    promts = load_promts(args.input)
    for p in promts:
        prompt_res = build_prompt(p.prompt, res)
        prompts.append(prompt_res)
    return prompts
