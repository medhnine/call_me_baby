import json
import sys
from typing import Any, Dict
from pydantic import BaseModel, ValidationError, model_validator


class ParameterType(BaseModel):
    type: str


class ReturnType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ReturnType


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def verfy_len(self) -> "Prompt":
        res = self.prompt.strip()
        if res is None or len(res) == 0:
            raise ValueError("empty prompt")
        return self


def load_promts(path: str) -> list[Prompt]:
    promts = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            pr = Prompt(**item)
            promts.append(pr)
    except ValidationError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    return promts


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    functions = []
    d1: dict[str, Any] = {
        "name": "fn_unknown",
        "description": (
            "Fallback function to use when no other function is "
            "appropriate for the user's prompt."
        ),
        "parameters": {},
        "returns": {
            "type": "string"
        }
    }
    try:
        with open(path, "r") as f:
            data = json.load(f)
        data.append(d1)
        for item in data:
            fn = FunctionDefinition(**item)
            functions.append(fn)
    except ValidationError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"an error has been aquired {e}")
        sys.exit(1)
    return functions
