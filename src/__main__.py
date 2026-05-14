"""Entry point for the call-me-maybe function calling system."""

import argparse
import sys
from typing import Optional


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



if __name__ == "__main__":
    main()