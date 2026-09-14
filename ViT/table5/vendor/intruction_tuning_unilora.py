# Code based on https://github.com/GraphPKU/PiSSA script with minimal changes.

import copy
import math
import os
import yaml
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
from transformers.trainer_pt_utils import get_parameter_names
import torch
import transformers
from transformers import Trainer, set_seed
from datasets import load_dataset
from peft import (
    get_peft_model,
    UniLoRAConfig,
    UniLoRARoSASnipConfig,
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dataset_split: str = field(
        default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    preprocessing_num_workers: int = field(
        default=1,
        metadata={
            "help": (
                "Number of dataset tokenization worker processes. Keep this low "
                "to avoid pyarrow/aws-c-io thread exhaustion at process shutdown."
            )
        },
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_r: int = field(
        default=None, metadata={"help": "The rank of incremental matrices."}
    )
    num_vectors: int = field(
        default=None,
        metadata={
            "help": "Number of vectors in the vector bank. Use higher values when the model size increases."
        },
    )
    vector_length: int = field(
        default=256,
        metadata={
            "help": "The length of the vectors in the vector bank. The length of the vectors should be divisible by the hidden dimension of the model."
        },
    )
    save_only_topk_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether to only save the topk weights. Setting save_only_topk_weights = True significantly reduces storage space. However, models saved in this mode can be used for merging or inference only, not for resuming training."
        },
    )
    unilora_variant: str = field(
        default="unilora",
        metadata={
            "help": (
                "Adapter variant. Supported by the multi-GPU entry point: "
                "lora, unilora, unilora_rosa_snip, unilora_rosa_snip_multi_gpu."
            )
        },
    )
    theta_d_length: int = field(
        default=524288,
        metadata={"help": "Trainable theta_d length for UniLoRA-RoSA-SNIP."},
    )
    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Uniform init bound for UniLoRA-RoSA-SNIP theta_d."},
    )
    rosa_density: float = field(
        default=0.0,
        metadata={"help": "Sparse compensation density for UniLoRA-RoSA-SNIP. Overridden by rosa_sparse_budget if set."},
    )
    rosa_sparse_budget: int = field(
        default=0,
        metadata={"help": "Exact sparse-position budget used to derive rosa_density when greater than zero."},
    )
    rosa_warmup_steps: int = field(
        default=128,
        metadata={"help": "Optimizer steps before collecting SNIP gradients for the RoSA sparse mask."},
    )
    rosa_mask_steps: int = field(
        default=1,
        metadata={"help": "Optimizer steps whose gradients are accumulated to select the RoSA sparse mask."},
    )
    learning_rate_vector_bank: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for UniLoRA vector bank. Defaults to learning_rate."},
    )
    learning_rate_theta_d: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for UniLoRA-RoSA-SNIP theta_d. Defaults to learning_rate_vector_bank."},
    )
    learning_rate_sparse: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for RoSA sparse vector. Defaults to theta_d lr * rosa_sparse_lr_mult."},
    )
    rosa_sparse_lr_mult: float = field(
        default=0.2,
        metadata={"help": "Sparse-vector LR multiplier when learning_rate_sparse is not specified."},
    )
    rosa_reset_optimizer_on_mask: bool = field(
        default=True,
        metadata={"help": "Clear optimizer state when the RoSA sparse mask is activated."},
    )
    rosa_decay_sparse_lr_after_activation: bool = field(
        default=True,
        metadata={"help": "Decay RoSA sparse LR after sparse-mask activation."},
    )
    rosa_fail_on_zero_scores: bool = field(
        default=True,
        metadata={"help": "Abort RoSA-SNIP training if mask saliency scores are all zero."},
    )
    rosa_fail_on_zero_sparse_update: bool = field(
        default=True,
        metadata={"help": "Abort RoSA-SNIP training if sparse parameters remain zero after activation."},
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device map used by from_pretrained. Use 'auto' to shard a large model across all visible GPUs, or 'none' to disable it."
        },
    )
    max_memory_per_gpu: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional per-GPU max_memory passed to from_pretrained, e.g. '44GiB'."
        },
    )
    max_memory_cpu: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional CPU max_memory passed to from_pretrained when max_memory_per_gpu is set, e.g. '128GiB'."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [
        PROMPT.format_map(dict(instruction=instruction))
        for instruction in examples[query]
    ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def resolve_torch_dtype(script_args):
    if script_args.bf16:
        return torch.bfloat16
    if script_args.fp16:
        return torch.float16
    return None


def build_max_memory(script_args):
    if script_args.max_memory_per_gpu is None:
        return None

    max_memory = {
        device_id: script_args.max_memory_per_gpu
        for device_id in range(torch.cuda.device_count())
    }
    if script_args.max_memory_cpu is not None:
        max_memory["cpu"] = script_args.max_memory_cpu
    return max_memory


def count_unilora_matrix_positions(args, model, target_modules):
    target_modules = set(target_modules)
    total_positions = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        module_name = name.split(".")[-1]
        if module_name == "lm_head" or module_name not in target_modules:
            continue
        total_positions += args.lora_r * module.in_features + module.out_features * args.lora_r
    return total_positions


def get_unilora_rosa_backend(model):
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)

    visited = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if all(
            hasattr(candidate, attr)
            for attr in (
                "should_collect_gradients",
                "should_generate_masks",
                "generate_sparse_masks",
            )
        ):
            return candidate
        candidates.append(getattr(candidate, "base_model", None))
        candidates.append(getattr(candidate, "model", None))
    return None


def get_theta_d_lr(args):
    vector_lr = args.learning_rate_vector_bank
    if vector_lr is None:
        vector_lr = args.learning_rate
    return args.learning_rate_theta_d if args.learning_rate_theta_d is not None else vector_lr


def resolve_rosa_sparse_lr(args):
    if args.learning_rate_sparse is not None:
        return args.learning_rate_sparse
    return get_theta_d_lr(args) * args.rosa_sparse_lr_mult


def get_rosa_sparse_lr_for_step(args, step_after_update, total_steps):
    base_lr = resolve_rosa_sparse_lr(args)
    if not args.rosa_decay_sparse_lr_after_activation:
        return base_lr

    activation_step = args.rosa_warmup_steps + args.rosa_mask_steps
    if step_after_update <= activation_step:
        return base_lr

    decay_steps = max(1, total_steps - activation_step)
    progress = (float(step_after_update) - float(activation_step)) / float(decay_steps)
    progress = max(0.0, min(1.0, progress))
    scheduler_name = str(args.lr_scheduler_type).lower()
    if "cosine" in scheduler_name:
        lr_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        lr_factor = 1.0 - progress
    return base_lr * max(0.0, lr_factor)


class UniLoRARoSATrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        rosa_backend = get_unilora_rosa_backend(model)
        rosa_collecting = False
        if rosa_backend is not None:
            rosa_collecting = rosa_backend.should_collect_gradients(self.state.global_step, adapter_name="default")
            rosa_backend.enable_gradient_capture(rosa_collecting, mode="snip")

        loss = super().training_step(model, inputs, *args, **kwargs)

        if rosa_backend is not None:
            if rosa_collecting:
                capture_stats = rosa_backend.accumulate_gradient_statistics(adapter_name="default")
                if capture_stats.get("updated_tensors", 0) == 0:
                    raise RuntimeError(
                        "UniLoRA-RoSA-SNIP did not capture any A/B gradients during the sparse-mask collection "
                        "window. Check gradient checkpointing and adapter gradient flow before continuing."
                    )
            else:
                rosa_backend.enable_gradient_capture(False)
        return loss


class UniLoRARoSACallback(transformers.TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        rosa_backend = get_unilora_rosa_backend(model)
        if rosa_backend is not None:
            print(
                "UniLoRA-RoSA-SNIP sparse config: "
                f"density={args.rosa_density}, "
                f"warmup_steps={args.rosa_warmup_steps}, "
                f"mask_steps={args.rosa_mask_steps}, "
                f"sparse_lr={resolve_rosa_sparse_lr(args)}, "
                f"reset_optimizer_on_mask={args.rosa_reset_optimizer_on_mask}"
            )
            print(f"Initial sparse stats: {rosa_backend.get_sparse_structure_stats()}")
        return control

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        rosa_backend = get_unilora_rosa_backend(model)
        if rosa_backend is None:
            return control

        if rosa_backend.should_generate_masks(state.global_step, adapter_name="default"):
            mask_info = rosa_backend.generate_sparse_masks(adapter_name="default")
            if args.rosa_fail_on_zero_scores and mask_info.get("score_max", 0.0) <= 0.0:
                raise RuntimeError(
                    "UniLoRA-RoSA-SNIP generated a sparse mask from zero saliency scores. "
                    "Aborting because the sparse adapter would remain inactive."
                )
            if optimizer is not None and args.rosa_reset_optimizer_on_mask:
                optimizer.state.clear()
            sparse_stats = {}
            if hasattr(rosa_backend, "get_sparse_parameter_stats"):
                sparse_stats = rosa_backend.get_sparse_parameter_stats(adapter_name="default")
            print(
                "Activated UniLoRA-RoSA-SNIP sparse compensation: "
                f"selected_ratio={mask_info['selected_ratio']:.6f}, "
                f"selected_positions={mask_info['selected_positions']}, "
                f"density={mask_info['selected_density']:.6f}, "
                f"score_max={mask_info['score_max']:.6e}, "
                f"selected_sparse_nonzero={sparse_stats.get('selected_sparse_nonzero', 0)}, "
                f"selected_sparse_absmax={sparse_stats.get('selected_sparse_absmax', 0.0):.6e}"
            )

        if optimizer is not None:
            sparse_lr = get_rosa_sparse_lr_for_step(args, state.global_step, state.max_steps)
            for group in optimizer.param_groups:
                if group.get("unilora_param_group") == "rosa_sparse":
                    group["lr"] = sparse_lr

        if (
            state.global_step > 0
            and state.global_step % 100 == 0
            and hasattr(rosa_backend, "has_sparse_masks")
            and rosa_backend.has_sparse_masks(adapter_name="default")
            and hasattr(rosa_backend, "get_sparse_parameter_stats")
        ):
            sparse_stats = rosa_backend.get_sparse_parameter_stats(adapter_name="default")
            activation_step = args.rosa_warmup_steps + args.rosa_mask_steps
            if (
                args.rosa_fail_on_zero_sparse_update
                and state.global_step > activation_step
                and sparse_stats["selected_sparse_nonzero"] == 0
            ):
                raise RuntimeError(
                    "UniLoRA-RoSA-SNIP sparse parameters are still all zero after mask activation. "
                    "Sparse gradients or optimizer updates are not flowing."
                )
            print(
                "UniLoRA-RoSA-SNIP sparse parameter stats: "
                f"step={state.global_step}, "
                f"sparse_nonzero={sparse_stats['sparse_nonzero']}, "
                f"sparse_absmax={sparse_stats['sparse_absmax']:.6e}, "
                f"selected_sparse_nonzero={sparse_stats['selected_sparse_nonzero']}, "
                f"selected_sparse_absmax={sparse_stats['selected_sparse_absmax']:.6e}"
            )
        return control


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

    if script_args.lora_r is not None:
        modules = find_all_linear_names(model)
        unilora_variant = script_args.unilora_variant.lower()
        if unilora_variant == "unilora_rosa_snip":
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
                "adding UniLoRA-RoSA-SNIP modules... "
                f"theta_d_length={script_args.theta_d_length}, "
                f"total_sparse_positions={total_sparse_positions}, "
                f"sparse_budget={script_args.rosa_sparse_budget}, "
                f"rosa_density={script_args.rosa_density}"
            )
            config = UniLoRARoSASnipConfig(
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
        elif unilora_variant == "unilora":
            print(f"adding Uni-LoRA modules...")
            config = UniLoRAConfig(
                r=script_args.lora_r,
                vector_length=script_args.vector_length,
                unilora_dropout=0,
                target_modules=modules,
                num_vectors=script_args.num_vectors,
                task_type="CAUSAL_LM",
                save_only_topk_weights=script_args.save_only_topk_weights,
            )
        else:
            raise ValueError(f"Unsupported unilora_variant: {script_args.unilora_variant}")

        model = get_peft_model(model, config)
        if script_args.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        raise ValueError("LoRA rank should be provided.")

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S-%f")

    adapter_name = "default"
    peft_config_dict = {adapter_name: config}

    run_name = f"rank_{peft_config_dict[adapter_name].r}_lr_{script_args.learning_rate}_seed_{script_args.seed}"
    if script_args.unilora_variant.lower() == "unilora_rosa_snip":
        run_name = (
            f"unilora_rosa_snip_td_{script_args.theta_d_length}_sb_{script_args.rosa_sparse_budget}_"
            f"w_{script_args.rosa_warmup_steps}_m_{script_args.rosa_mask_steps}_"
            f"rank_{peft_config_dict[adapter_name].r}_lr_{script_args.learning_rate}_seed_{script_args.seed}"
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

    raw_train_datasets = load_dataset(
        script_args.data_path, split=script_args.dataset_split
    )
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
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    optimizer = create_optimizer(model, script_args)
    trainer_cls = UniLoRARoSATrainer if script_args.unilora_variant.lower() == "unilora_rosa_snip" else Trainer
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        optimizers=(optimizer, None),
        **data_module,
    )
    if script_args.unilora_variant.lower() == "unilora_rosa_snip":
        trainer.add_callback(UniLoRARoSACallback)

    print_trainable_parameters(model)

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, "ft"))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if "logits" in name:
                num = param.numel() / param.shape[-1] * 3
            else:
                num = param.numel()
            print(name, param.dtype, param.shape, num)
            trainable_params += num
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def create_optimizer(model, args) -> torch.optim.Optimizer:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    vector_bank_parameters = [
        name for name, _ in model.named_parameters() if "vector_bank" in name
    ]
    logits_parameters = [
        name for name, _ in model.named_parameters() if "logits" in name
    ]
    rosa_theta_d_parameters = [
        name for name, _ in model.named_parameters() if "unilora_rosa_theta_d" in name
    ]
    rosa_sparse_parameters = [
        name for name, _ in model.named_parameters() if "unilora_rosa_sparse_theta_D" in name
    ]
    special_lr_parameters = set(
        logits_parameters + vector_bank_parameters + rosa_theta_d_parameters + rosa_sparse_parameters
    )

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters
                and n not in special_lr_parameters
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters
                and n not in special_lr_parameters
            ],
            "weight_decay": 0.0,
        },
    ]
    if vector_bank_parameters:
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in model.named_parameters() if n in vector_bank_parameters
                ],
                "lr": args.learning_rate_vector_bank or args.learning_rate,
                "weight_decay": 0.0,
                "unilora_param_group": "vector_bank",
            }
        )
    if rosa_theta_d_parameters:
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in model.named_parameters() if n in rosa_theta_d_parameters
                ],
                "lr": get_theta_d_lr(args),
                "weight_decay": 0.0,
                "unilora_param_group": "rosa_theta_d",
            }
        )
    if rosa_sparse_parameters:
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in model.named_parameters() if n in rosa_sparse_parameters
                ],
                "lr": resolve_rosa_sparse_lr(args),
                "weight_decay": 0.0,
                "unilora_param_group": "rosa_sparse",
            }
        )

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


if __name__ == "__main__":
    train()
