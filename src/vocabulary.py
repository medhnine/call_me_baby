import os
os.environ.setdefault("HF_HOME", "/goinfre/mohhnine/.cache/huggingface")

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]  # noqa: E402


def load_vocabulary() -> None:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        result = f.read()
    print(result)
    ids_lis = model.encode("what is the capital of paris ?")
    re1 = model.get_logits_from_input_ids(ids_lis[0].tolist())
    re = model.decode(max(re1[0]))
    print(re)


load_vocabulary()
