"""Evaluate a merged model on the 8 commonsense benchmarks from LLM-Adapters
(BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA).

Test files are the LLM-Adapters dataset/<name>/test.json files (list of dicts
with "instruction" and "answer"). The prompt template matches training
(alpaca no-input template) and answers are extracted with the same regexes as
LLM-Adapters' commonsense_evaluate.py.

Example:
    python instruction_tuning_eval/commonsense_eval.py \
        --model output_merged/... \
        --dataset boolq \
        --data_file data/commonsense/boolq_test.json
"""

import argparse
import json
import re

from vllm import LLM, SamplingParams

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

EXTRACT_PATTERNS = {
    "boolq": r"true|false",
    "piqa": r"solution1|solution2",
    "social_i_qa": r"answer1|answer2|answer3",
    "hellaswag": r"ending1|ending2|ending3|ending4",
    "winogrande": r"option1|option2",
    "ARC-Challenge": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Easy": r"answer1|answer2|answer3|answer4|answer5",
    "openbookqa": r"answer1|answer2|answer3|answer4",
}


def extract_answer(dataset: str, sentence: str) -> str:
    matches = re.findall(EXTRACT_PATTERNS[dataset], sentence)
    return matches[0] if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(EXTRACT_PATTERNS))
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional json file for per-example predictions.")
    args = parser.parse_args()

    with open(args.data_file, encoding="utf8") as f:
        examples = json.load(f)

    prompts = [PROMPT.format(instruction=ex["instruction"]) for ex in examples]
    answers = [str(ex["answer"]).strip() for ex in examples]
    print(f"dataset={args.dataset} total={len(prompts)}")

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=args.max_new_tokens,
        stop=["### Instruction", "Instruction:"],
    )
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    outputs = llm.generate(prompts, sampling_params)
    # vLLM preserves input order in returned outputs.
    completions = [out.outputs[0].text for out in outputs]

    records = []
    n_correct = 0
    n_invalid = 0
    for ex, completion, answer in zip(examples, completions, answers):
        pred = extract_answer(args.dataset, completion)
        if not pred:
            n_invalid += 1
        correct = pred == answer
        n_correct += int(correct)
        records.append(
            {
                "instruction": ex["instruction"],
                "completion": completion,
                "pred": pred,
                "answer": answer,
                "correct": correct,
            }
        )

    acc = n_correct / len(records)
    print(f"invalid_predictions==== {n_invalid}")
    print(f"{args.dataset} length==== {len(records)} , {args.dataset} acc==== {acc:.6f}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
