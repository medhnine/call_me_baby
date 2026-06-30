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
                return (self.model.decode(fn_ids[0][:-1]))
            if not fn_ids:
                raise ValueError(f"No surviving function after pruning at position {pos}")
            pos += 1
    
    def force_tokens(self, tokens : str, input_ids):
        ids = self.model.encode(tokens).tolist()[0]
        input_ids.extend(ids)

    def state_force(self, data:str, input_ids):
        length = len(data)
        count = 0
        if data[length - 1] != '}':
            count += 1
        if data[length - 2] != '}':
            count += 1
        for _ in range(0, count):
            self.force_tokens('}', input_ids)
            
    def generate_number(self, input_ids, point):
        result = []
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids + result)
            next_token = logits.index(max(logits))
            if ',' in self.model.decode([next_token]) or '}' in self.model.decode([next_token]):
                res = self.model.decode(result)
                num = ''
                try:
                    for chr in res:
                        if chr in '.-0123456789':
                            num += chr
                    fl = num
                    if point == False:
                        fl = float(num)
                except Exception as e:
                    print(e)
                    return
                input_ids.extend(self.model.encode(' ' + str(fl)).tolist()[0])
                return fl
            else:
                result.append(next_token)

    def generate_string(self, input_ids, boolean):
        result = []
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids + result)
            # if boolean:
            #     allowd = self.model.encode("True").tolist()[0]
            #     allowd += self.model.encode("False").tolist()[0]
            #     allowd += self.model.encode(" false").tolist()[0]
            #     allowd = self.model.encode(" true").tolist()[0]
            #     allowd += self.model.encode('"').tolist()[0]
            #     next_token = max(allowd, key=lambda log : logits[log])
            # else:
            next_token = logits.index(max(logits))
            if '"' in self.model.decode(next_token):
                if len(self.model.decode(next_token)) == 1:
                        return result
                else:
                    data : str = self.model.decode(next_token)
                    i = data.index('"')
                    r = data[:i]
                    result.extend(self.model.encode(r).tolist()[0])
                    return result
                if len(self.model.decode(next_token)):
                    return result
            else:
                result.append(next_token)
 
    def generate_paramters(self, fn_name, fns_obj, input_ids):
        for fn in fns_obj:
            if fn.name == fn_name:
                pos = 0
                output = {}
                for key , val in fn.parameters.items():
                    res = '"' + key
                    if pos + 1 <= len(fn.parameters):
                        res += '"'
                    self.force_tokens(res, input_ids)
                    if val.type in ["number", "integer", "float"]:
                        self.force_tokens(':', input_ids)
                        if val.type == "integer":
                            output[key] = int(self.generate_number(input_ids, True))
                        else:
                            output[key] = float(self.generate_number(input_ids, False))
                        if pos < len(fn.parameters) - 1:
                            input_ids.append(self.model.encode(', ').tolist()[0][0])
                            input_ids.append(self.model.encode(' ').tolist()[0][0])
                        if pos == len(fn.parameters) - 1:
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                    if val.type in ["string", "boolean"]:
                        self.force_tokens(': ', input_ids)
                        input_ids.append(self.model.encode('"').tolist()[0][0])
                        if val.type == "boolean":
                            res = self.generate_string(input_ids, True)
                        else:
                            res = self.generate_string(input_ids, False)
                        output[key] = self.model.decode(res)
                        input_ids.extend(res)
                        input_ids.append(self.model.encode('"').tolist()[0][0])
                        if pos < len(fn.parameters) - 1:
                            input_ids.append(self.model.encode(',').tolist()[0][0])
                            input_ids.append(self.model.encode(' ').tolist()[0][0])
                        if pos == len(fn.parameters) - 1:
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                    pos += 1
                return output
