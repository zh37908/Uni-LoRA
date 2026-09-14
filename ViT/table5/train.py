"""One isolated ProLoSA trial. Tuning never opens the test split."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_from_disk
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainerCallback, set_seed
from peft import UniLoRARoSASnipConfig, get_peft_model
from intruction_tuning_unilora import TrainingArguments, UniLoRARoSATrainer, UniLoRARoSACallback
from prepare import atomic_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tunable(model):
    return {n:p for n,p in model.named_parameters() if "unilora_rosa_theta_d" in n or "unilora_rosa_sparse_theta_D" in n or "classifier" in n}


class BestValidation(TrainerCallback):
    def __init__(self, output, spec):
        self.output, self.spec = output, spec
        self.best = None
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if "eval_accuracy" not in metrics:
            return
        score, loss = float(metrics["eval_accuracy"]), float(metrics["eval_loss"])
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss")
        row = dict(epoch=state.epoch, step=state.global_step, accuracy=score, loss=loss)
        self.history.append(row)
        if self.best is None or (score, -loss) > (self.best["accuracy"], -self.best["loss"]):
            self.best = row
            checkpoint = dict(spec=self.spec, best=row,
                              parameters={n:p.detach().cpu().clone() for n,p in tunable(model).items()},
                              sparse_mask=model.base_model.unilora_rosa_sparse_mask["default"].detach().cpu().clone())
            temp = self.output / "best.tmp"
            torch.save(checkpoint, temp)
            temp.replace(self.output / "best.pt")
        atomic_json(self.output / "validation_history.json", self.history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    opts = parser.parse_args()
    spec = read(opts.spec)
    out = Path(opts.output)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "spec.json", spec)
    start = time.time()
    set_seed(spec["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one allocated CUDA GPU")
    dataset = spec["dataset"]
    meta = read(ROOT / "data" / dataset / "manifest.json")
    model_meta = read(ROOT / "models" / f'{spec["model"]}.json')
    train = load_from_disk(str(ROOT / "data" / dataset / "train"))
    val = load_from_disk(str(ROOT / "data" / dataset / "validation"))
    if spec["stage"] == "smoke":
        train = train.shuffle(seed=101).select(range(min(512, len(train))))
        val = val.select(range(min(256, len(val))))
    processor = AutoImageProcessor.from_pretrained(model_meta["path"], local_files_only=True)
    size = processor.size["height"]
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    train_tf = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    eval_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def train_transform(batch):
        return {"pixel_values": [train_tf(x.convert("RGB")) for x in batch["image"]], "label": batch["label"]}

    def eval_transform(batch):
        return {"pixel_values": [eval_tf(x.convert("RGB")) for x in batch["image"]], "label": batch["label"]}

    def collate(examples):
        return dict(pixel_values=torch.stack([x["pixel_values"] for x in examples]),
                    labels=torch.tensor([x["label"] for x in examples], dtype=torch.long))

    train.set_transform(train_transform)
    val.set_transform(eval_transform)
    classes = meta["classes"]
    backbone = AutoModelForImageClassification.from_pretrained(
        model_meta["path"], local_files_only=True, ignore_mismatched_sizes=True,
        label2id={n:i for i,n in enumerate(classes)}, id2label={i:n for i,n in enumerate(classes)},
        attn_implementation="sdpa")
    dim, depth = backbone.config.hidden_size, backbone.config.num_hidden_layers
    full_dim = depth * 2 * 2 * dim * 4
    budget = 72000 if spec["model"] == "base" else 144000
    # Nearest integer to B/(ratio+1); exact residual count is checked after activation.
    sparse = round(budget / (spec["ratio"] + 1))
    latent = budget - sparse
    micro = spec.get("micro_batch", 64 if spec["model"] == "base" else 32)
    effective = 512
    assert effective % micro == 0
    accumulation = effective // micro
    steps_per_epoch = math.ceil(math.ceil(len(train) / micro) / accumulation)
    steps = steps_per_epoch * spec["epochs"]
    warmup = max(1, int(spec.get("support_warmup_ratio", 0.1) * steps))
    if spec["stage"] == "smoke": warmup = 0
    # Keep ceil(D*density) numerically equal to K in the existing sparse backend.
    density = (sparse - 0.25) / full_dim
    config = UniLoRARoSASnipConfig(r=4, theta_d_length=latent, proj_seed=spec["seed"],
                                  init_theta_d_bound=0.02, unilora_dropout=0,
                                  rosa_density=density, rosa_warmup_steps=warmup,
                                  rosa_mask_steps=1, target_modules=["query", "value"])
    model = get_peft_model(backbone, config)
    for n,p in model.named_parameters():
        if "classifier" in n: p.requires_grad_(True)
    head = [p for n,p in model.named_parameters() if "classifier" in n]
    z = [p for n,p in model.named_parameters() if "unilora_rosa_theta_d" in n]
    residual = [p for n,p in model.named_parameters() if "unilora_rosa_sparse_theta_D" in n]
    assert sum(p.numel() for p in z) == latent
    assert sum(p.numel() for p in residual) == full_dim
    sparse_mult = spec.get("sparse_lr_mult", 0.2)
    optimizer = torch.optim.AdamW([
        dict(params=head, lr=spec["head_lr"], weight_decay=0.01),
        dict(params=z, lr=spec["vector_lr"], weight_decay=0.01, unilora_param_group="rosa_theta_d"),
        dict(params=residual, lr=spec["vector_lr"] * sparse_mult, weight_decay=0.01, unilora_param_group="rosa_sparse"),
    ], betas=(0.9, 0.999), eps=1e-8)
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=spec["epochs"], per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro, gradient_accumulation_steps=accumulation,
        learning_rate=spec["vector_lr"], learning_rate_theta_d=spec["vector_lr"],
        rosa_sparse_lr_mult=sparse_mult, rosa_warmup_steps=warmup, rosa_mask_steps=1,
        rosa_density=density, rosa_sparse_budget=sparse, rosa_reset_optimizer_on_mask=True,
        rosa_decay_sparse_lr_after_activation=True, eval_strategy="epoch", save_strategy="no",
        lr_scheduler_type="linear", warmup_ratio=0.0, weight_decay=0.01,
        fp16=True, bf16=False, logging_steps=max(1, steps_per_epoch),
        dataloader_num_workers=4, dataloader_persistent_workers=True,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        seed=spec["seed"], data_seed=spec["seed"], disable_tqdm=True,
    )
    # Match the legacy scheduler (zero LR warmup); support warmup is separate.
    callback = BestValidation(out, spec)
    trainer = UniLoRARoSATrainer(
        model=model, args=args, optimizers=(optimizer, None), train_dataset=train,
        eval_dataset=val, data_collator=collate,
        compute_metrics=lambda p: {"accuracy": float((np.argmax(p.predictions, axis=-1) == p.label_ids).mean())},
        callbacks=[UniLoRARoSACallback, callback],
    )
    print("TRIAL", json.dumps(dict(spec=spec, d=latent, K=sparse, D=full_dim, effective_batch=effective,
                                    train=len(train), validation=len(val), support_warmup=warmup,
                                    gpu=torch.cuda.get_device_name(0))), flush=True)
    trained = trainer.train()
    stats = model.base_model.get_sparse_structure_stats()
    pstats = model.base_model.get_sparse_parameter_stats()
    assert stats["selected_positions"] == sparse, stats
    assert pstats["selected_sparse_nonzero"] > 0, pstats
    assert callback.best is not None
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for n,p in tunable(model).items(): p.copy_(state["parameters"][n])
        model.base_model.unilora_rosa_sparse_mask["default"].copy_(state["sparse_mask"])
    model.base_model._sync_sparse_requires_grad_with_masks()
    if spec["stage"] == "smoke":
        restored = trainer.evaluate(val, metric_key_prefix="restore")
        assert abs(restored["restore_accuracy"] - callback.best["accuracy"]) < 1e-12
    result = dict(spec=spec, spec_sha256=digest(spec), best_val_accuracy=callback.best["accuracy"],
                  best_val_loss=callback.best["loss"], best_epoch=callback.best["epoch"],
                  train_runtime_sec=trained.metrics["train_runtime"], d=latent, K=sparse,
                  adapter_active_dof=budget, head_params=sum(p.numel() for p in head),
                  total_active_dof=budget+sum(p.numel() for p in head),
                  sparse_activation=stats, sparse_parameter_stats=pstats,
                  data_manifest_sha256=meta["manifest_sha256"], model_revision=model_meta["revision"],
                  micro_batch=micro, gradient_accumulation=accumulation, effective_batch=effective,
                  precision="fp16", weight_decay=0.01, scheduler="linear", scheduler_warmup_ratio=0,
                  support_warmup_steps=warmup, gpu=torch.cuda.get_device_name(0),
                  training_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    # No tuning/smoke path loads, evaluates, or reports test data.
    if spec["stage"] == "final":
        test = load_from_disk(str(ROOT / "data" / dataset / "test"))
        test.set_transform(eval_transform)
        test_metrics = trainer.evaluate(test, metric_key_prefix="test")
        result.update(test_accuracy=test_metrics["test_accuracy"], test_loss=test_metrics["test_loss"], test_size=len(test))
    result["wall_runtime_sec"] = time.time() - start
    result["checkpoint_sha256"] = hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()
    atomic_json(out / "result.json", result)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
