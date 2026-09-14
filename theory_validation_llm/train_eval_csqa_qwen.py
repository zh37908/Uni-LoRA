"""P4 theory-alignment experiments on a real LLM (Qwen2.5-1.5B + CommonsenseQA).

E1 (sample-size sweep):   --train_fraction p, fixed number of update steps.
E2 (label-noise sweep):   --label_noise eta, full training set, labels flipped
                          uniformly to a wrong choice with a fixed noise seed.

Mirrors the setup of "LoRA vs. Full Fine-Tuning: A Theoretical Perspective"
(Qwen2.5 + CommonsenseQA, noise/sample sweeps) but compares
LoRA vs Uni-LoRA vs ProLoSA under matched budgets.

The training subset (per fraction) and the noisy labels (per eta) are decided
by dedicated seeds (subset_seed / noise_seed) that stay fixed across methods
and training seeds, so all methods see exactly the same data.

Evaluation: multiple-choice accuracy on the CommonsenseQA validation split by
per-choice log-likelihood scoring with the (in-memory) PEFT model.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import Trainer, set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "math_instruction_tuning"))

from intruction_tuning_unilora import (  # noqa: E402
    PROMPT,
    DataCollatorForSupervisedDataset,
    TrainingArguments,
    UniLoRARoSACallback,
    UniLoRARoSATrainer,
    create_optimizer,
    find_all_linear_names,
    print_trainable_parameters,
    train_tokenize_function,
)
from intruction_tuning_unilora_multi_gpu import build_peft_config  # noqa: E402
from peft import get_peft_model  # noqa: E402

ANSWER_TEMPLATE = "the correct answer is answer{idx}"


@dataclass
class TheoryArguments:
    experiment: str = field(default="e1", metadata={"help": "e1 (sample size) or e2 (label noise)"})
    train_fraction: float = field(default=1.0, metadata={"help": "E1: fraction of the training set."})
    label_noise: float = field(default=0.0, metadata={"help": "E2: label flip probability."})
    subset_seed: int = field(default=1234, metadata={"help": "Subset sampling seed (fixed across methods)."})
    noise_seed: int = field(default=5678, metadata={"help": "Label-noise seed (fixed across methods)."})
    eval_batch_size: int = field(default=16)
    result_file: str = field(default=None)


def format_instruction(example):
    lines = [f"Question: {example['question']}", "Choose the correct answer from the following options:"]
    for i, text in enumerate(example["choices"]["text"]):
        lines.append(f"answer{i + 1}: {text}")
    return "\n".join(lines)


def build_examples(theory_args):
    raw = load_dataset("tau/commonsense_qa", split="train")
    labels = []
    for ex in raw:
        labels.append(ex["choices"]["label"].index(ex["answerKey"]))

    n = len(raw)
    indices = np.arange(n)
    if theory_args.train_fraction < 1.0:
        rng = np.random.RandomState(theory_args.subset_seed)
        take = max(1, int(round(theory_args.train_fraction * n)))
        indices = np.sort(rng.choice(n, size=take, replace=False))

    noisy = 0
    rng_noise = np.random.RandomState(theory_args.noise_seed)
    records = []
    for idx in indices:
        ex = raw[int(idx)]
        label = labels[int(idx)]
        num_choices = len(ex["choices"]["text"])
        if theory_args.label_noise > 0 and rng_noise.rand() < theory_args.label_noise:
            wrong = [k for k in range(num_choices) if k != label]
            label = int(rng_noise.choice(wrong))
            noisy += 1
        records.append(
            {
                "instruction": format_instruction(ex),
                "output": ANSWER_TEMPLATE.format(idx=label + 1),
            }
        )
    print(f"train examples: {len(records)} (flipped labels: {noisy})")
    return records


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, eval_batch_size):
    """Accuracy on CommonsenseQA validation via per-choice log-likelihood."""
    val = load_dataset("tau/commonsense_qa", split="validation")
    device = next(model.parameters()).device
    model.eval()

    # Flatten (example, choice) pairs.
    prompts, completions, owners = [], [], []
    gold = []
    for i, ex in enumerate(val):
        gold.append(ex["choices"]["label"].index(ex["answerKey"]))
        source = PROMPT.format_map({"instruction": format_instruction(ex)})
        for k in range(len(ex["choices"]["text"])):
            prompts.append(source)
            completions.append(ANSWER_TEMPLATE.format(idx=k + 1))
            owners.append(i)

    scores = []
    for start in range(0, len(prompts), eval_batch_size):
        batch_prompts = prompts[start : start + eval_batch_size]
        batch_completions = completions[start : start + eval_batch_size]
        input_ids_list, labels_list = [], []
        for p, c in zip(batch_prompts, batch_completions):
            p_ids = tokenizer(p, return_tensors="pt").input_ids[0]
            c_ids = tokenizer(c, add_special_tokens=False, return_tensors="pt").input_ids[0]
            ids = torch.cat([p_ids, c_ids])
            lab = ids.clone()
            lab[: len(p_ids)] = -100
            input_ids_list.append(ids)
            labels_list.append(lab)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        labels_pad = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        ).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels_pad[:, 1:]
        logprobs = torch.log_softmax(shift_logits, dim=-1)
        mask = shift_labels.ne(-100)
        token_lp = logprobs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        scores.extend((token_lp * mask).sum(dim=-1).tolist())

    # argmax over each example's choices
    per_example = {}
    for owner, score in zip(owners, scores):
        per_example.setdefault(owner, []).append(score)
    correct = sum(int(int(np.argmax(v)) == gold[i]) for i, v in per_example.items())
    acc = correct / len(per_example)
    print(f"csqa validation: n={len(per_example)} acc={acc:.6f}")
    return acc


def main():
    parser = transformers.HfArgumentParser((TheoryArguments, TrainingArguments))
    theory_args, script_args = parser.parse_args_into_dataclasses()
    print(theory_args)
    print(script_args)

    if theory_args.experiment == "e1":
        assert theory_args.label_noise == 0.0, "E1 uses clean labels."
    set_seed(script_args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if script_args.bf16 else None,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    modules = find_all_linear_names(model)
    config = build_peft_config(script_args, model, modules)
    model = get_peft_model(model, config)
    if script_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    records = build_examples(theory_args)
    raw_ds = Dataset.from_list(records)
    train_dataset = raw_ds.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        remove_columns=raw_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer, "query": "instruction", "response": "output"},
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    optimizer = create_optimizer(model, script_args)
    is_rosa = "rosa" in script_args.unilora_variant.lower()
    trainer_cls = UniLoRARoSATrainer if is_rosa else Trainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        optimizers=(optimizer, None),
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if is_rosa:
        trainer.add_callback(UniLoRARoSACallback)

    print_trainable_parameters(model)
    train_result = trainer.train()

    model.config.use_cache = True
    acc = evaluate_multiple_choice(model, tokenizer, theory_args.eval_batch_size)

    result = {
        "experiment": theory_args.experiment,
        "model": script_args.model_name_or_path,
        "method": script_args.unilora_variant,
        "train_fraction": theory_args.train_fraction,
        "label_noise": theory_args.label_noise,
        "seed": script_args.seed,
        "subset_seed": theory_args.subset_seed,
        "noise_seed": theory_args.noise_seed,
        "max_steps": script_args.max_steps,
        "accuracy": acc,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "gpu_peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available()
        else None,
        "gpu_peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3)
        if torch.cuda.is_available()
        else None,
    }
    print("RESULT:", json.dumps(result))
    if theory_args.result_file:
        os.makedirs(os.path.dirname(theory_args.result_file), exist_ok=True)
        with open(theory_args.result_file, "w", encoding="utf8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
