"""One-configuration PEFT baselines; reuse the existing math data protocol.

Use NLU/peft/src, which already contains VB-LoRA, VeRA and FourierFT.
No adapter implementation is changed. FourierFT has no LoRA rank.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
import peft
from datasets import load_dataset
from peft import VBLoRAConfig, VeraConfig, FourierFTConfig, get_peft_model
from intruction_tuning_unilora import (
    TrainingArguments, DataCollatorForSupervisedDataset, train_tokenize_function,
    find_all_linear_names,
)

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


@dataclass
class BaselineArguments(TrainingArguments):
    method: str = 'vblora'
    trainable_budget: int = 1048576
    learning_rate_logits: float = 0.01
    fourier_scaling: float = 300.0
    fourier_target_scope: str = 'all'


def target_modules(args):
    if args.fourier_target_scope not in ('all', 'qv'):
        raise ValueError('fourier_target_scope must be all or qv')
    if args.method == 'fourierft' and args.fourier_target_scope == 'qv':
        return ['q_proj', 'v_proj']
    return TARGETS


def build_config(model, args):
    targets = target_modules(args)
    modules = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in targets]
    if not modules:
        raise ValueError('No target modules found')
    common = dict(target_modules=targets, task_type='CAUSAL_LM', bias='none')
    if args.method == 'vblora':
        config = VBLoRAConfig(r=args.lora_r, num_vectors=args.num_vectors,
                              vector_length=args.vector_length, topk=2,
                              save_only_topk_weights=False, vblora_dropout=0, **common)
        expected = args.num_vectors * (args.vector_length + sum(
            args.lora_r * (m.in_features + m.out_features) // args.vector_length
            for _, m in modules))
    elif args.method == 'vera':
        config = VeraConfig(r=args.lora_r, d_initial=0.1, vera_dropout=0,
                            save_projection=True, projection_prng_key=args.seed, **common)
        expected = sum(m.out_features + args.lora_r for _, m in modules)
    elif args.method == 'fourierft':
        n, rem = divmod(args.trainable_budget, len(modules))
        # Full module paths allow exact global budget allocation.
        pattern = {name: n + 1 for name, _ in sorted(modules)[:rem]}
        config = FourierFTConfig(n_frequency=n, n_frequency_pattern=pattern,
                                 scaling=args.fourier_scaling, random_loc_seed=args.seed,
                                 init_weights=True, **common)
        expected = args.trainable_budget
    else:
        raise ValueError(f'Unsupported existing baseline: {args.method}')
    return config, expected


def build_optimizer(model, args):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        kind = 'logits' if 'vblora_logits' in name else 'vector_bank' if 'vblora_vector_bank' in name else 'adapter'
        groups.setdefault(kind, []).append(param)
    rates = dict(logits=args.learning_rate_logits,
                 vector_bank=args.learning_rate_vector_bank or args.learning_rate,
                 adapter=args.learning_rate)
    return torch.optim.AdamW([
        dict(params=params, lr=rates[kind], weight_decay=args.weight_decay, group_name=kind)
        for kind, params in groups.items()
    ], betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)


def train():
    args = transformers.HfArgumentParser(BaselineArguments).parse_args_into_dataclasses()[0]
    if torch.cuda.device_count() != 1:
        raise RuntimeError('This experiment requires exactly one visible GPU')
    if not args.gradient_checkpointing:
        raise ValueError('Gradient checkpointing must be enabled')
    if args.lora_r != 4:
        raise ValueError('This comparison fixes the LoRA-based methods at rank 4')
    output = Path(args.output_dir)
    if (output / 'ft').exists():
        raise FileExistsError(f'Refusing to overwrite completed adapter: {output}')
    transformers.set_seed(args.seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
        device_map={'': 0}, low_cpu_mem_usage=True,
    )
    if set(find_all_linear_names(model)) != set(TARGETS):
        raise ValueError('Backbone targets differ from the existing Uni-LoRA experiment')
    config, expected = build_config(model, args)
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.config.use_cache = False
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(f'Trainable parameter mismatch: {actual} != {expected}')
    output.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, args)
    manifest = dict(method=args.method, model=args.model_name_or_path, seed=args.seed,
                    rank=None if args.method == 'fourierft' else args.lora_r,
                    target_budget=args.trainable_budget, actual_trainable_params=actual,
                    peft_source=peft.__file__, target_modules=target_modules(args),
                    gpu=torch.cuda.get_device_name(0), gpu_count=1,
                    optimizer_groups=[dict(name=g['group_name'], lr=g['lr'],
                                           numel=sum(p.numel() for p in g['params']))
                                      for g in optimizer.param_groups])
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'training_arguments.json').write_text(args.to_json_string())
    print('RUN_MANIFEST:', json.dumps(manifest), flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.model_max_length,
        padding_side='right', use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    raw = load_dataset(args.data_path, split=args.dataset_split)
    data = raw.map(train_tokenize_function, batched=True, batch_size=3000,
                   num_proc=args.preprocessing_num_workers, remove_columns=raw.column_names,
                   load_from_cache_file=True,
                   fn_kwargs=dict(tokenizer=tokenizer, query=args.dataset_field[0],
                                  response=args.dataset_field[1]))
    trainer = transformers.Trainer(
        model=model, tokenizer=tokenizer, args=args, optimizers=(optimizer, None),
        train_dataset=data, data_collator=DataCollatorForSupervisedDataset(tokenizer))
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    model.save_pretrained(output / 'ft')
    tokenizer.save_pretrained(output / 'ft')
    resource = dict(result.metrics, gpu_peak_alloc_GiB=torch.cuda.max_memory_allocated()/2**30,
                    gpu_peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    (output / 'resource_usage.json').write_text(json.dumps(resource, indent=2) + '\n')
    (output / 'TRAINING_COMPLETE').write_text('Training and adapter save succeeded.\n')
    print('TRAINING_COMPLETE:', output, flush=True)
    try:
        from pyarrow import fs
        fs.finalize_s3()
    except (ImportError, AttributeError):
        pass


if __name__ == '__main__':
    train()
