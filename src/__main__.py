"""Entry point for the call-me-maybe function calling system."""

import argparse
import json
import os

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .decoder import ConstrainedDecoder
from .function_caller import function_caller
from .parser import load_function_definitions, load_promts
from .promt_builder import function_names

os.makedirs("data/output", exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling system using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to output JSON file."
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline."""
    args = parse_arguments()
    print(f"Functions definition: {args.functions_definition}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    prompts = function_caller(args)
    user_pormts = load_promts(args.input)
    function_definitions = load_function_definitions(args.functions_definition)
    fn_names = function_names(function_definitions)
    if not prompts:
        print("No prompts found")
        return
    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(model)

    list_objects = []
    for i, x in zip(prompts, user_pormts):
        input_ids = model.encode(i).tolist()[0]
        add = model.encode('{"name": "').tolist()[0]
        input_ids.extend(add)
        try:
            generated_name = decoder.generate_function_name(
                input_ids, fn_names
            )
            target = (
                f', "prompt": "{x.prompt}", '
            )
            decoder.force_tokens(target, input_ids)
            decoder.force_tokens('"parameters": {', input_ids)
            parm = decoder.generate_paramters(
                generated_name, function_definitions, input_ids
            )
        except Exception as e:
            print(f"error {e}")
            exit(1)
        res = {
            "prompt": x.prompt,
            "name": generated_name,
            "parameters": parm
            }
        list_objects.append(res)
    with open(args.output, "w") as f:
        json.dump(list_objects, f, indent=4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
