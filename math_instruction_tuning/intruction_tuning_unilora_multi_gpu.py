import datetime
import json
import os

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, UniLoRAConfig, UniLoRARoSASnipConfig, UniLoRARoSASnipMultiGpuConfig, get_peft_model
from transformers import Trainer, set_seed

from intruction_tuning_unilora import (
    DataCollatorForSupervisedDataset,
    TrainingArguments,
    UniLoRARoSACallback,
    UniLoRARoSATrainer,
    build_max_memory,
    count_unilora_matrix_positions,
    create_optimizer,
    find_all_linear_names,
    print_trainable_parameters,
    resolve_torch_dtype,
    train_tokenize_function,
)


ROSA_SNIP_VARIANTS = {
    "unilora_rosa_snip": UniLoRARoSASnipConfig,
    "unilora_rosa_snip_multi_gpu": UniLoRARoSASnipMultiGpuConfig,
}


def build_peft_config(script_args, model, modules):
    unilora_variant = script_args.unilora_variant.lower()
    if unilora_variant == "lora":
        print("adding standard LoRA modules...")
        return LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            lora_dropout=0,
            target_modules=modules,
            task_type="CAUSAL_LM",
        )

    if unilora_variant in ROSA_SNIP_VARIANTS:
        total_sparse_positions = count_unilora_matrix_positions(script_args, model, modules)
        if total_sparse_positions <= 0:
            raise ValueError("No target modules found for UniLoRA-RoSA-SNIP sparse position counting.")
        if script_args.rosa_sparse_budget > 0:
            if script_args.rosa_sparse_budget > total_sparse_positions:
                raise ValueError(
                    f"rosa_sparse_budget={script_args.rosa_sparse_budget} exceeds "
                    f"total sparse positions={total_sparse_positions}."
                )
            script_args.rosa_density = script_args.rosa_sparse_budget / total_sparse_positions
        print(
            f"adding {unilora_variant} modules... "
            f"theta_d_length={script_args.theta_d_length}, "
            f"total_sparse_positions={total_sparse_positions}, "
            f"sparse_budget={script_args.rosa_sparse_budget}, "
            f"rosa_density={script_args.rosa_density}"
        )
        config_cls = ROSA_SNIP_VARIANTS[unilora_variant]
        return config_cls(
            r=script_args.lora_r,
            theta_d_length=script_args.theta_d_length,
            proj_seed=script_args.seed,
            init_theta_d_bound=script_args.init_theta_d_bound,
            unilora_dropout=0,
            rosa_density=script_args.rosa_density,
            rosa_warmup_steps=script_args.rosa_warmup_steps,
            rosa_mask_steps=script_args.rosa_mask_steps,
            target_modules=modules,
            task_type="CAUSAL_LM",
        )

    if unilora_variant == "unilora":
        print("adding Uni-LoRA modules...")
        return UniLoRAConfig(
            r=script_args.lora_r,
            vector_length=script_args.vector_length,
            unilora_dropout=0,
            target_modules=modules,
            num_vectors=script_args.num_vectors,
            task_type="CAUSAL_LM",
            save_only_topk_weights=script_args.save_only_topk_weights,
        )

    raise ValueError(
        f"Unsupported unilora_variant: {script_args.unilora_variant}. "
        "Supported: lora, unilora, unilora_rosa_snip, unilora_rosa_snip_multi_gpu."
    )


def collect_resource_usage(train_result, script_args):
    """Wall time / throughput / per-GPU peak memory, written next to the checkpoint."""
    metrics = getattr(train_result, "metrics", {}) or {}
    usage = {
        "model": script_args.model_name_or_path,
        "variant": script_args.unilora_variant,
        "seed": script_args.seed,
        "train_runtime_sec": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        "num_gpus": torch.cuda.device_count(),
        "gpu_peak_alloc_GiB": {},
        "gpu_peak_reserved_GiB": {},
    }
    for i in range(torch.cuda.device_count()):
        usage["gpu_peak_alloc_GiB"][f"cuda:{i}"] = round(torch.cuda.max_memory_allocated(i) / 2**30, 3)
        usage["gpu_peak_reserved_GiB"][f"cuda:{i}"] = round(torch.cuda.max_memory_reserved(i) / 2**30, 3)
    return usage


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    set_seed(script_args.seed)

    device_map = script_args.device_map
    if device_map is not None and device_map.lower() == "none":
        device_map = None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map=device_map,
        max_memory=build_max_memory(script_args),
        torch_dtype=resolve_torch_dtype(script_args),
        low_cpu_mem_usage=True,
    )
    if script_args.lora_r is None:
        raise ValueError("LoRA rank should be provided.")

    modules = find_all_linear_names(model)
    config = build_peft_config(script_args, model, modules)

    model = get_peft_model(model, config)
    if script_args.gradient_checkpointing:
        model.enable_input_require_grads()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S-%f")
    adapter_name = "default"
    peft_config_dict = {adapter_name: config}

    run_name = f"rank_{peft_config_dict[adapter_name].r}_lr_{script_args.learning_rate}_seed_{script_args.seed}"
    if script_args.unilora_variant.lower() in ROSA_SNIP_VARIANTS:
        run_name = (
            f"{script_args.unilora_variant.lower()}_td_{script_args.theta_d_length}_"
            f"sb_{script_args.rosa_sparse_budget}_w_{script_args.rosa_warmup_steps}_"
            f"m_{script_args.rosa_mask_steps}_rank_{peft_config_dict[adapter_name].r}_"
            f"lr_{script_args.learning_rate}_seed_{script_args.seed}"
        )

    script_args.output_dir = (
        f"{script_args.output_dir}/{script_args.model_name_or_path}/"
        f"{script_args.data_path}_split_{script_args.dataset_split}/"
        f"{run_name}/output_{now}"
    )
    os.makedirs(script_args.output_dir, exist_ok=True)

    for param in model.parameters():
        param.data = param.data.contiguous()

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if script_args.data_path.endswith((".json", ".jsonl")):
        # Local instruction-tuning file, e.g. Commonsense-170K from LLM-Adapters.
        raw_train_datasets = load_dataset(
            "json", data_files=script_args.data_path, split=script_args.dataset_split
        )
    else:
        raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=script_args.preprocessing_num_workers,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    optimizer = create_optimizer(model, script_args)
    trainer_cls = UniLoRARoSATrainer if script_args.unilora_variant.lower() in ROSA_SNIP_VARIANTS else Trainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        optimizers=(optimizer, None),
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if script_args.unilora_variant.lower() in ROSA_SNIP_VARIANTS:
        trainer.add_callback(UniLoRARoSACallback)

    print_trainable_parameters(model)

    train_result = trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, "ft"))

    resource = collect_resource_usage(train_result, script_args)
    print("RESOURCE:", json.dumps(resource))
    with open(os.path.join(script_args.output_dir, "resource_usage.json"), "w", encoding="utf8") as f:
        json.dump(resource, f, indent=1)

    # PyArrow 9 can abort during interpreter shutdown while its bundled AWS
    # event loop tries to create a cleanup thread. Finalize it while Python is
    # still fully running; this process performs no filesystem work afterwards.
    try:
        from pyarrow import fs as pyarrow_fs

        pyarrow_fs.finalize_s3()
        print("Finalized PyArrow S3 runtime.")
    except (ImportError, AttributeError):
        pass


if __name__ == "__main__":
    train()
