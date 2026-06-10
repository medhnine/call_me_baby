from llm_sdk import Small_LLM_Model

class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
    def filter_logits(self, input_ids: list[int], store: set[int]) -> int:
        logits: list = self.model.get_logits_from_input_ids(input_ids)
        return max(store, key=lambda log : logits[log])

    def generate_function_name(self, input_ids: list[int], function_names: list[str]) -> str:
        pos = 0
        fn_ids = []
        for fn in function_names:
            token_list = self.model.encode(fn).tolist()[0]
            fn_ids.append(token_list)
        while True:
            store = {item[pos] for item in fn_ids if pos < len(item)}
            store.add(self.model.encode('"').tolist()[0][0])
            next_token = self.filter_logits(input_ids, store)
            input_ids.append(next_token)
            fn_ids = [fn for fn in fn_ids if pos < len(fn) and fn[pos] == next_token]
            if self.model.decode([next_token]) == '"':
                return (self.model.decode(fn_ids[0]))
            if not fn_ids:
                raise ValueError(f"No surviving function after pruning at position {pos}")
            pos += 1
    def generate_paramters(self, functions_obj, input_ids, user_promt):
        fn
        pass
        




























































    # def generate_function_name(self, input_ids: list[int], function_names: list[str]) -> str:
    #     fn_ids = {}
    #     name = ""
    #     for fn in function_names:
    #         fn_ids[fn] = self.model.encode(fn).tolist()[0]
    #     pos = 0
    #     while True:
    #         store = {i[pos] for i in fn_ids.values() if pos < len(i)}
    #         next_id = self.filter_logits(input_ids, store)
    #         input_ids.append(next_id)
    #         fn_ids = {fn : ids for fn, ids in fn_ids.items() if pos < len(ids) and next_id == ids[pos]}
    #         pos += 1
    #         if len(fn_ids) == 1:
    #             name = next(iter(fn_ids))
    #             if pos == len(fn_ids[name]):
    #                 return name
    #         elif len(fn_ids) == 0:
    #             raise ValueError(f"No surviving function after pruning at position {pos}")

