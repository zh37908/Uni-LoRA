"""P1 vision experiment: ViT-B/16 few-shot / full-data image classification
with LoRA / Uni-LoRA / ProLoSA (UniLoRA-RoSA-SNIP) under matched budgets.

Reuses the RoSA-SNIP trainer machinery and TrainingArguments from
math_instruction_tuning/intruction_tuning_unilora.py so that ProLoSA behaves
identically to the language experiments.

Example:
    python train_vit_p1.py \
        --unilora_variant unilora --dataset cifar100 --train_size 1000 \
        --model_name_or_path google/vit-base-patch16-224-in21k \
        --vector_length 24600 --lora_r 4 --learning_rate 4e-3 --head_lr 3e-3 \
        --num_train_epochs 20 --seed 42 --output_dir output/p1_debug ...
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    set_seed,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "math_instruction_tuning"))

from intruction_tuning_unilora import (  # noqa: E402
    TrainingArguments,
    UniLoRARoSACallback,
    UniLoRARoSATrainer,
    count_unilora_matrix_positions,
    get_theta_d_lr,
    resolve_rosa_sparse_lr,
)
from peft import (  # noqa: E402
    LoraConfig,
    UniLoRAConfig,
    UniLoRARoSASnipConfig,
    get_peft_model,
)

# name -> (hf_repo, image_key, label_key, train_split, test_split)
DATASETS = {
    "cifar100": ("uoft-cs/cifar100", "img", "fine_label", "train", "test"),
    "food101": ("ethz/food101", "image", "label", "train", "validation"),
    "dtd": ("tanganke/dtd", "image", "label", "train", "test"),
}

TARGET_MODULES = ["query", "value"]


@dataclass
class VisionArguments:
    dataset: str = field(default="cifar100")
    train_size: int = field(
        default=1000, metadata={"help": "Stratified train subset size; -1 uses the full split."}
    )
    subset_seed: int = field(
        default=42, metadata={"help": "Seed for subset sampling, kept fixed across methods/seeds."}
    )
    head_lr: float = field(default=3e-3, metadata={"help": "Learning rate of the classifier head."})
    result_file: str = field(default=None, metadata={"help": "Where to write the result json."})


def stratified_indices(labels, train_size, subset_seed):
    """Per-class balanced subset of `train_size` indices."""
    rng = np.random.RandomState(subset_seed)
    by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)
    classes = sorted(by_class)
    base = train_size // len(classes)
    remainder = train_size - base * len(classes)
    selected = []
    for i, cls in enumerate(rng.permutation(classes)):
        take = base + (1 if i < remainder else 0)
        pool = by_class[cls]
        take = min(take, len(pool))
        selected.extend(rng.choice(pool, size=take, replace=False))
    return sorted(int(i) for i in selected)


def build_peft_config(script_args, model):
    variant = script_args.unilora_variant.lower()
    if variant == "lora":
        return LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            lora_dropout=0,
            target_modules=TARGET_MODULES,
        )
    if variant == "unilora":
        return UniLoRAConfig(
            r=script_args.lora_r,
            vector_length=script_args.vector_length,
            unilora_dropout=0,
            target_modules=TARGET_MODULES,
            num_vectors=1,
        )
    if variant == "unilora_rosa_snip":
        total_positions = count_unilora_matrix_positions(script_args, model, TARGET_MODULES)
        if script_args.rosa_sparse_budget > 0:
            script_args.rosa_density = script_args.rosa_sparse_budget / total_positions
        print(
            f"ProLoSA config: theta_d={script_args.theta_d_length}, "
            f"sparse_budget={script_args.rosa_sparse_budget}, "
            f"total_positions={total_positions}, density={script_args.rosa_density}"
        )
        return UniLoRARoSASnipConfig(
            r=script_args.lora_r,
            theta_d_length=script_args.theta_d_length,
            proj_seed=script_args.seed,
            init_theta_d_bound=script_args.init_theta_d_bound,
            unilora_dropout=0,
            rosa_density=script_args.rosa_density,
            rosa_warmup_steps=script_args.rosa_warmup_steps,
            rosa_mask_steps=script_args.rosa_mask_steps,
            target_modules=TARGET_MODULES,
        )
    raise ValueError(f"Unsupported unilora_variant: {script_args.unilora_variant}")


def create_optimizer_vit(model, args, head_lr):
    """Param groups: classifier head / vector bank / theta_d / rosa sparse / rest."""
    head_params, vector_bank, theta_d, rosa_sparse, rest = [], [], [], [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            head_params.append(param)
        elif "unilora_rosa_sparse_theta_D" in name:
            # Sparse params start frozen (requires_grad=False) and are unfrozen at
            # mask activation, so they must be registered by name, not by requires_grad.
            rosa_sparse.append(param)
        elif not param.requires_grad:
            continue
        elif "unilora_rosa_theta_d" in name:
            theta_d.append(param)
        elif "vector_bank" in name:
            vector_bank.append(param)
        else:
            rest.append(param)

    groups = [{"params": head_params, "lr": head_lr, "weight_decay": 0.0}]
    if rest:
        groups.append({"params": rest, "lr": args.learning_rate, "weight_decay": 0.0})
    if vector_bank:
        groups.append(
            {
                "params": vector_bank,
                "lr": args.learning_rate_vector_bank or args.learning_rate,
                "weight_decay": 0.0,
                "unilora_param_group": "vector_bank",
            }
        )
    if theta_d:
        groups.append(
            {
                "params": theta_d,
                "lr": get_theta_d_lr(args),
                "weight_decay": 0.0,
                "unilora_param_group": "rosa_theta_d",
            }
        )
    if rosa_sparse:
        groups.append(
            {
                "params": rosa_sparse,
                "lr": resolve_rosa_sparse_lr(args),
                "weight_decay": 0.0,
                "unilora_param_group": "rosa_sparse",
            }
        )
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    optimizer_kwargs.pop("lr", None)
    return optimizer_cls(groups, **optimizer_kwargs)


def main():
    parser = transformers.HfArgumentParser((VisionArguments, TrainingArguments))
    vision_args, script_args = parser.parse_args_into_dataclasses()
    print(vision_args)
    print(script_args)

    set_seed(script_args.seed)

    repo, image_key, label_key, train_split, test_split = DATASETS[vision_args.dataset]
    train_ds = load_dataset(repo, split=train_split)
    test_ds = load_dataset(repo, split=test_split)

    label_names = train_ds.features[label_key].names
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}

    if vision_args.train_size > 0:
        indices = stratified_indices(train_ds[label_key], vision_args.train_size, vision_args.subset_seed)
        train_ds = train_ds.select(indices)
        print(f"Selected {len(train_ds)} training examples (subset_seed={vision_args.subset_seed}).")

    image_processor = AutoImageProcessor.from_pretrained(script_args.model_name_or_path)
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

    size = image_processor.size["height"]
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_tf = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def preprocess_train(batch):
        batch["pixel_values"] = [train_tf(img.convert("RGB")) for img in batch[image_key]]
        return batch

    def preprocess_val(batch):
        batch["pixel_values"] = [val_tf(img.convert("RGB")) for img in batch[image_key]]
        return batch

    train_ds.set_transform(preprocess_train)
    test_ds.set_transform(preprocess_val)

    model = AutoModelForImageClassification.from_pretrained(
        script_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model = get_peft_model(model, build_peft_config(script_args, model))

    # Keep the classification head trainable for all methods.
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "classifier" in n)
    print(f"trainable params: {trainable} (classifier head: {head}, adapter: {trainable - head})")

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex[label_key] for ex in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": float((preds == eval_pred.label_ids).mean())}

    script_args.label_names = ["labels"]
    script_args.remove_unused_columns = False

    optimizer = create_optimizer_vit(model, script_args, vision_args.head_lr)
    is_rosa = script_args.unilora_variant.lower() == "unilora_rosa_snip"
    trainer_cls = UniLoRARoSATrainer if is_rosa else Trainer
    trainer = trainer_cls(
        model=model,
        args=script_args,
        optimizers=(optimizer, None),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    if is_rosa:
        trainer.add_callback(UniLoRARoSACallback)

    train_result = trainer.train()

    final_metrics = trainer.evaluate(test_ds)
    eval_accs = [
        entry["eval_accuracy"] for entry in trainer.state.log_history if "eval_accuracy" in entry
    ]
    result = {
        "dataset": vision_args.dataset,
        "method": script_args.unilora_variant,
        "seed": script_args.seed,
        "train_size": vision_args.train_size,
        "subset_seed": vision_args.subset_seed,
        "final_accuracy": final_metrics["eval_accuracy"],
        "best_accuracy": max(eval_accs) if eval_accs else final_metrics["eval_accuracy"],
        "trainable_params": trainable,
        "adapter_params": trainable - head,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "gpu_peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available()
        else None,
        "gpu_peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3)
        if torch.cuda.is_available()
        else None,
    }
    print("RESULT:", json.dumps(result))
    if vision_args.result_file:
        os.makedirs(os.path.dirname(vision_args.result_file), exist_ok=True)
        with open(vision_args.result_file, "w", encoding="utf8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
