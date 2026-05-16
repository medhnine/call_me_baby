from typing import Any
from llm_sdk import Small_LLM_Model

class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model, vocabulary: dict[int, str]):
        self.model = model
        self.vocabulary = vocabulary
    def filter_logits(self, input_ids: list[int], store: set[int]) -> int:
        logits: list = self.model.get_logits_from_input_ids(input_ids)
        return (max(store, key=lambda i: logits[i]))

    def generate_function_name(self, input_ids: list[int], function_names: list[str]) -> str:
        fn_ids = {}
        name = ""
        for fn in function_names:
            fn_ids[fn] = self.model.encode(fn).tolist()[0]
        pos = 0
        while True:
            store = {i[pos] for i in fn_ids.values() if pos < len(i)}
            next_id = self.filter_logits(input_ids, store)
            input_ids.append(next_id)
            fn_ids = {fn : ids for fn, ids in fn_ids.items() if pos < len(ids) and next_id == ids[pos]}
            pos += 1
            if len(fn_ids) == 1:
                name = next(iter(fn_ids))
                if pos == len(fn_ids[name]):
                    return name
            elif len(fn_ids) == 0:
                raise ValueError(f"No surviving function after pruning at position {pos}")

