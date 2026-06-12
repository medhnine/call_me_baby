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
            fn_ids = [fn for fn in fn_ids if pos < len(fn) and fn[pos] == next_token]
            input_ids.append(next_token)
            if self.model.decode([next_token]) == '"':
                print(self.model.decode(fn_ids[0]))
                return (self.model.decode(fn_ids[0][:-1]))
            if not fn_ids:
                raise ValueError(f"No surviving function after pruning at position {pos}")
            pos += 1
    
    def force_tokens(self, tokens : str, input_ids):
        ids = self.model.encode(tokens).tolist()[0]
        input_ids.extend(ids)

    # def generate_number(self, input_ids):
    #     allowed_chrs = '-01234.56789,}"'
    #     digits_ids = self.model.encode("0123456789.").tolist()[0]
    #     number = []
    #     allowed = []
    #     for i in allowed_chrs:
    #         allowed.append(self.model.encode(i).tolist()[0][0])
       
    #     next_token = self.model.encode('1').tolist()[0][0]
    #     pos = 0
    #     while next_token not in self.model.encode(',}').tolist()[0]:
    #         logits = self.model.get_logits_from_input_ids(input_ids)
    #         next_token = max(allowed, key=lambda log : logits[log])
    #         if next_token in digits_ids:
    #             number.append(next_token)
    #             input_ids.append(next_token)
    #         else:
    #             input_ids.append(next_token)
    #         if self.model.decode([next_token]) in ',}':
    #             result = self.model.decode(number)
    #             try:
    #                 ret = result
    #                 result = float(result)
    #                 result = str(result)
    #                 return number
    #             except Exception as e:
    #                 print(e)
    #             return 
    #         pos += 1

    def generate_number(self, input_ids):
            allowed_chrs = '0123456789,}."'
            stop_chrs = self.model.encode(',}').tolist()[0]
            allowed = self.model.encode(allowed_chrs).tolist()[0]
            next_token = self.model.encode('+').tolist()[0][0]
            ids_copy = input_ids.copy()
            pos = 0
            result = []
            while True:
                if next_token in self.model.encode("-0123456789.").tolist()[0]:
                    result.append(next_token)
                if next_token in stop_chrs:
                    number = float(self.model.decode(result))
                    return self.model.encode(str(number)).tolist()[0]
                if pos == 0:
                    digits_ids = self.model.encode("0123456789").tolist()[0]
                    minsign_id = self.model.encode('-').tolist()[0][0]
                    alloweds = digits_ids + [minsign_id]
                    logits = self.model.get_logits_from_input_ids(ids_copy)
                    print(f"Logits for minus sign: {logits[minsign_id]}")
                    next_token = max(alloweds, key=lambda log : logits[log])
                    print(f"Logits for next token: {logits[next_token]}")
                else:
                    logits = self.model.get_logits_from_input_ids(ids_copy)
                    next_token = max(allowed, key=lambda log : logits[log])
                ids_copy.append(next_token)
                pos += 1

    def generate_paramters(self, fn_name, fns_obj, input_ids):
        for fn in fns_obj:
            if fn.name == fn_name:
                pos = 0
                for key , val in fn.parameters.items():
                    self.force_tokens('"' + key + '"' if pos < len(fn.parameters) - 1 else None, input_ids)
                    self.force_tokens(': ', input_ids)
                    if val.type == 'number':
                        input_ids.extend(self.generate_number(input_ids))
                    elif val.type == 'string':
                        pass
            break





























































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

