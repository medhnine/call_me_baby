"""Entry point for the call-me-maybe function calling system."""

import argparse
from .parser import load_function_definitions, load_promts
from .function_caller import function_caller
from .decoder import ConstrainedDecoder
from llm_sdk import Small_LLM_Model
from .promt_builder import function_names

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
    function_definitions = load_function_definitions(args.functions_definition)
    fn_names = function_names(function_definitions)
    print(fn_names)
    if not prompts:
        print("No prompts found")
        return
    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(model)
    inputs_id = model.encode(prompts[0]).tolist()[0]
    for i in prompts:
        inputs_id = model.encode(i).tolist()[0]
        generated_name = decoder.generate_function_name(inputs_id, fn_names)
        print(generated_name)
if __name__ == "__main__":
    main()
