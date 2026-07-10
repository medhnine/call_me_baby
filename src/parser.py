import json
import sys
from typing import Dict
from pydantic import BaseModel, ValidationError

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

def load_promts(path:str)-> list[Prompt]:
    promts = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            pr = Prompt(**item)
            promts.append(pr)
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

def load_function_definitions(path:str)-> list[FunctionDefinition]:
    functions = []
    d1: dict = {
    "name": "fn_unknown",
    "description": "Fallback function to use when no other function is appropriate for the user's prompt.",
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