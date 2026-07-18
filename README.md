*This project has been created as part of the 42 curriculum by mohhnine.*

<!-- Replace/append logins above if this was a group project: by <login1>, <login2>, ... -->

# Call Me Maybe — Function Calling in LLMs

## Description

This project turns a natural-language request into a **structured function call** instead of a plain answer.

Given a prompt like `"What is the sum of 40 and 2?"`, the program does **not** return `42`. It returns a machine-readable instruction describing *which* function should be called and *with what arguments*:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The point is that an LLM can only produce text — it cannot run code, read files, or fetch live data. Function calling lets the model produce a precise instruction that a *real program* can then execute. Addition is a weak example (the model could just answer it); the real value shows up for tasks the model physically cannot do, such as `fn_read_file` or `fn_execute_sql_query`.

The hard requirement is that the output be **100% valid JSON that matches the given schema, every single time** — because whatever consumes this output breaks on a single malformed result. A small model (Qwen3-0.6B, ~600M parameters) is unreliable at producing valid JSON from prompting alone (often ~30% of the time). This project reaches 100% validity using **constrained decoding**, not prompting.

## Instructions

### Requirements
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk/` package (copied into the repo, provides the `Small_LLM_Model` wrapper around Qwen3-0.6B)

### Install
```bash
uv sync
```
This installs `numpy` and `pydantic` and wires up the local `llm_sdk` workspace member.

### Run
Default paths (reads from `data/input/`, writes to `data/output/`):
```bash
uv run python -m src
```

Custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile shortcuts
- `make install` — install dependencies (`uv sync`)
- `make run` — run the program
- `make debug` — run under the Python debugger (pdb)
- `make lint` — run `flake8` and `mypy`
- `make clean` — remove `__pycache__` and `.mypy_cache`

## How it works (algorithm)

### The core idea: constrained decoding

An LLM generates one token at a time. At each step it outputs a **logit** (a raw score) for every token in its vocabulary — roughly 150,000 scores. Normally you take the highest-scoring token and hope the result is valid.

Prompting can only make valid output *more likely*; it can never make it *certain*, because any nonzero chance of a bad token eventually fires. Constrained decoding solves this differently: at each step it **restricts which tokens are allowed** and takes the highest-scoring token **only among the legal ones**. An illegal token, however high the model scores it, is never in the candidate set — so it cannot be selected. Validity is therefore guaranteed **by construction**, not by luck.

The model is never modified or retrained. We control only the selection step that happens *after* the model produces its scores.

### Three generation strategies

"Valid output" is not one problem — it splits into three, by argument type:

1. **Function name — a finite set.** There is a fixed list of real function names. Each name is encoded into its token path, and generation walks position by position: at each step only the tokens that continue one of the surviving names are allowed, pruning candidates until a single name completes. The model can never invent a function name, because invented names are not in the candidate set. Selection only matters at the positions where names actually differ; shared prefixes are effectively forced.

2. **Number argument — a rule-based (regular) language.** There is no finite list of numbers, so the legal set is defined by type rules: digits, an optional minus, an optional dot. Generation stops when the model reaches a `,` or `}` (the JSON boundary). Integer vs float is resolved by casting the decoded value (`int(...)` / `float(...)`).

3. **String argument — free text.** A string can contain almost anything, so generation is unconstrained argmax over all tokens, with the **closing quote** enforced as the terminator.

### Structural termination

The loop does not wait for the model's end-of-sequence (EOS) token. Instead, **the JSON structure decides when to stop**: a name ends at its closing quote, a number ends at a comma/brace, a string ends at its closing quote. Because we know the shape of a valid answer, completion is detected structurally rather than relying on the model.

### The role of the prompt

A prompt containing the available functions and the user request is fed to the model. This affects **accuracy only** (it helps the model prefer the *right* function), never **validity** (which is guaranteed by the constraint). Removing the prompt would lower accuracy but leave validity at 100% — proof that the two come from different places.

### The `fn_unknown` fallback

Because the constraint forces the model to pick *some* function, a nonsense prompt (e.g. `"apples are wonderfll"`) would otherwise be forced onto a wrong function. A synthetic `fn_unknown` function (zero parameters) is added as a legal candidate, giving the constraint a valid "none of these" escape hatch.

## Design decisions

- **Restrict the candidate set instead of masking to −infinity.** The textbook formulation sets illegal logits to `-inf`, applies softmax, then samples. This project instead takes the argmax over the legal set directly. The result is identical (softmax is monotonic and only the max is needed) but avoids building a 150,000-element array at every step.
- **Greedy selection, no sampling.** This is deterministic extraction, not creative generation, so temperature / top-k / top-p are not used. Greedy argmax is the correct and reproducible choice.
- **pydantic for input validation.** Function definitions and prompts are parsed into pydantic models, so malformed input fails early, loudly, and in one place instead of causing a mysterious error deep in the pipeline.
- **Encode name + closing quote together.** Candidate names are encoded as `fn_name"` in one string so the closing-quote token sits naturally at the end of each token path. Encoding the name and quote separately could tokenize differently.

## Performance analysis

- **Validity: 100%.** Every output is well-formed, parseable JSON matching the schema, because invalid tokens are unreachable by construction.
- **Accuracy: ~90% (model-dependent).** Choosing the *right* function and extracting the *right* arguments depends on the model's preference within the legal set. A 0.6B model is small, so a fraction of answers are wrong — but even wrong answers are *valid* (a real function name in perfect JSON), so nothing downstream crashes. Accuracy is independent of the constraint layer: swapping in a stronger model raises accuracy with no code changes.
- **Speed.** All provided test prompts are processed well within the 5-minute target on standard hardware, since each generation step is a single forward pass and constraint checking is cheap.
- **Reliability.** Deterministic (greedy) selection means identical output across runs for the same input.

## Challenges faced

- **Token vs character mismatch.** The constraint reasons in characters ("the next character must be a digit"), while the model reasons in tokens (which may span several characters and encode leading spaces via the `Ġ` marker). Aligning the two — for example, gluing the closing quote onto each function name before encoding — was the subtle part.
- **Enforcing integer vs float.** The number constraint currently allows a decimal point and resolves the integer/float distinction *after* decoding via casting, rather than enforcing it inside the candidate set. A more rigorous version would drop the dot from the legal set for integer fields so the constraint itself makes a decimal point impossible.
- **Detecting the end of a string value.** Strings are generated freely, so the closing quote had to be detected even when it appears glued inside a larger token; the token is sliced at the quote to recover the exact string.
- **Unmatched prompts.** A constrained decoder must pick a function, so the `fn_unknown` fallback was added to avoid forcing wrong answers on prompts that match nothing.

## Testing strategy

- **Valid JSON check.** Every output file is re-parsed with `json.load` to confirm it is 100% parseable.
- **Schema conformance.** Each entry is checked for exactly the keys `prompt`, `name`, `parameters`, with argument names and types matching the function definitions.
- **Edge cases.** Tested with nonsense prompts (→ `fn_unknown`), large numbers, decimals, integer vs float fields, strings containing special characters, and functions with multiple parameters.
- **Error handling.** Tested with missing input files and malformed JSON to confirm the program fails gracefully with a clear message and never crashes.

## Example usage

Input (`data/input/function_calling_tests.json`):
```json
[
  { "prompt": "What is the product of 12 and 4?" },
  { "prompt": "Is 7 an even number?" },
  { "prompt": "apples are wonderfll" }
]
```

Run:
```bash
uv run python -m src
```

Output (`data/output/function_calling_results.json`):
```json
[
  { "prompt": "What is the product of 12 and 4?", "name": "fn_multiply_numbers", "parameters": { "a": 12.0, "b": 4.0 } },
  { "prompt": "Is 7 an even number?", "name": "fn_is_even", "parameters": { "n": 7 } },
  { "prompt": "apples are wonderfll", "name": "fn_unknown", "parameters": {} }
]
```

## Resources

- Qwen3-0.6B model — https://huggingface.co/Qwen/Qwen3-0.6B
- Byte-Pair Encoding (tokenization) — original BPE tokenization background
- Constrained / grammar-based decoding — the general technique behind reliable structured output (as used by tools such as `outlines` and llama.cpp GBNF grammars; these libraries are **not** used here — the mechanism is implemented from scratch)
- pydantic — https://docs.pydantic.dev/
- uv — https://docs.astral.sh/uv/

### Use of AI

AI was used as a **learning and review aid**, not to generate the implementation. Specifically:
- To explain the underlying concepts (tokenization/BPE, embeddings and position vectors, logits vs probabilities, the transformer pipeline, and constrained decoding) so the design could be reasoned about and defended.
- To review the constrained-decoding logic and identify weaknesses (e.g. integer/float enforcement, string termination edge cases, and error-handling gaps).
- All code was written, understood, and can be explained by the author; AI output was checked and validated rather than copied blindly.