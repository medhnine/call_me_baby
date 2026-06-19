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
    def generate_number(self, input_ids):
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
                    fl = float(num)
                except Exception as e:
                    print(e)
                    return
                input_ids.extend(self.model.encode(' ' + str(fl)).tolist()[0])
                return fl
            else:
                result.append(next_token)

    def generate_string(self, input_ids):
        result = []
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids + result)
            next_token = logits.index(max(logits))
            if '"' in self.model.decode(next_token):
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
                    if val.type == 'number':
                        self.force_tokens(':', input_ids)
                        output[key] = self.generate_number(input_ids)
                        if pos < len(fn.parameters) - 1:
                            input_ids.append(self.model.encode(', ').tolist()[0][0])
                            input_ids.append(self.model.encode(' ').tolist()[0][0])
                        if pos == len(fn.parameters) - 1:
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                            input_ids.append(self.model.encode('}').tolist()[0][0])
                    if val.type == 'string':
                        self.force_tokens(': ', input_ids)
                        input_ids.append(self.model.encode('"').tolist()[0][0])
                        res = self.generate_string(input_ids)
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


    # def decode_number(client: LLMClient, prompt_context_ids: list[int]) -> list[int]:
    #     digit_tokens = {client.encode(d)[0] for d in "0123456789."}
    #     comma = client.encode(",")[0]
    #     brace = client.encode("}")[0]
    #     minus = client.encode(" -")[0]
    #     allowed = digit_tokens | {comma, brace, minus}
    #     valueids = []
    #     for  in range(35):
    #         logits = client.get_logits(prompt_context_ids + value_ids)
    #         for index in range(len(logits)):
    #             if index not in allowed:
    #                 logits[index] = float("-inf")
    #         next_token = logits.index(max(logits))
    #         print(f'this score of minus{logits[minus]}')
    #         print(f'this score of next_token{logits[next_token]}')
    #         if next_token in (comma, brace):
    #             break
    #         value_ids.append(next_token)
    #     return value_ids
    
    # def generate_number(self, input_ids):
    #         digits = self.model.encode('0123456789').tolist()[0]
    #         stop_chrs = self.model.encode(',}').tolist()[0]
    #         point = self.model.encode('.').tolist()[0]
    #         minus = self.model.encode(' -').tolist()[0]
    #         allowed = []
    #         allowed += digits  + stop_chrs + minus + point
    #         while True:
    #             for x in range(5):
    #                 logits = self.model.get_logits_from_input_ids(input_ids)
    #                 if x == 0:
    #                     for log in range(len(logits)):
    #                         if logits.index(logits[log]) not in allowed:
    #                             logits[log] = float("-inf")
    #                 next_id = max(allowed, key=lambda log : logits[log])
    #                 print(f'score of minus {logits[minus[0]]}')
    #                 print(f'score of next_id {logits[next_id]}')
    #                 print()
    #                 input_ids.append(next_id)
    #                 if next_id in stop_chrs:
    #                     return
    #             break

    
    
































# def generate_number(self, input_ids):
#             allowed_chrs = '-0123456789,}."'
#             stop_chrs = self.model.encode(',}').tolist()[0]
#             allowed = self.model.encode(allowed_chrs).tolist()[0]
#             next_token = self.model.encode('+').tolist()[0][0]
#             ids_copy = input_ids.copy()
#             pos = 0
#             result = []
#             while True:
#                 logits = self.model.get_logits_from_input_ids(input_ids)
#                 next_token = max(allowed, key=lambda log : logits[log])
#                 input_ids.append(next_token)
#                 # if next_token in self.model.encode(".0123456789").tolist()[0]:
#                 #     result.append(next_token)
#                 if next_token in stop_chrs:
#                     # number = float(self.model.decode(result))
#                     # res = self.model.encode(str(number)).tolist()[0]
#                     # res.append(next_token)
#                     return
#                 # if pos == 0:
#                 #     digits_ids = self.model.encode("0123456789").tolist()[0]
#                 #     minsign_id = self.model.encode(' -').tolist()[0][0]
#                 #     alloweds = digits_ids + [minsign_id]
#                 #     logits = self.model.get_logits_from_input_ids(ids_copy)
#                 #     next_token = max(alloweds, key=lambda log : logits[log])
#                 #     if next_token == minsign_id:
#                 #         res = self.model.encode(' -').tolist()[0][0]
#                 #         result.append(res)
#                 #         ids_copy.append(res)
#                 # else:
#                 # logits = self.model.get_logits_from_input_ids(ids_copy)
#                 # next_token = max(allowed, key=lambda log : logits[log])
#                 # print(f'this is the value {self.model.decode([next_token])}')
#                 # print(f'score d point {logits[self.model.encode('.').tolist()[0][0]]}')
#                 # print(f'score d nexttoken {logits[next_token]}')

#                 # ids_copy.append(next_token)
#                 # pos += 1