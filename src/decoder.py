class ConstrainedDecoder:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
def filter_logits(self, input_ids, store):
    logits: list = self.model.get_logits_from_input_ids(input_ids)
    for token_id, _ in enumerate(logits):
        if token_id in store:
            continue
        else:
            logits[token_id] = float("-inf")
    return (logits.index(max(logits)))

def generate_function_name(self, input_ids: list[int], function_names: list[str]) -> str:
    remaining = function_names.copy()
    fn_ids = {}
    for fn in function_names:
        fn_ids[fn] = self.model.encode(fn).tolist()[0]
    pos = 0
    while True:
        store = [i[pos] for i in fn_ids.values()]
        next_id = filter_logits(input_ids, store)
        input_ids.append(next_id)
        pose += 1
        # flter function  that s noe anymore part of our chosen