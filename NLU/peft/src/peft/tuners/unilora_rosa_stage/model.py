from __future__ import annotations

import math

from .config import UniLoRARoSAStageConfig
from ..unilora_rosa.layer import UniLoRARoSALayer
from ..unilora_rosa.model import UniLoRARoSAModel


class UniLoRARoSAStageModel(UniLoRARoSAModel):
    """
    UniLoRA-RoSA variant whose sparse stage is triggered by training-epoch ratio.
    """

    prefix: str = "unilora_rosa_stage_"
    tuner_layer_cls = UniLoRARoSALayer

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        self._rosa_stage_start_steps: dict[str, int] = {}
        self._rosa_stage_schedule_meta: dict[str, dict[str, float | int]] = {}
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def set_training_schedule(
        self,
        total_steps: int,
        steps_per_epoch: int,
        total_epochs: int,
        stage_start_step_override: int | None = None,
        adapter_name: str = "default",
    ) -> dict[str, float | int]:
        config: UniLoRARoSAStageConfig = self.peft_config[adapter_name]
        total_steps = max(1, int(total_steps))
        steps_per_epoch = max(1, int(steps_per_epoch))
        total_epochs = max(1, int(total_epochs))

        using_warmup_steps = stage_start_step_override is not None
        if using_warmup_steps:
            stage_start_step = max(0, min(total_steps, int(stage_start_step_override)))
            stage_progress_epochs = float(stage_start_step) / float(steps_per_epoch)
            stage_ratio = float(stage_start_step) / float(total_steps)
        else:
            stage_progress_epochs = float(config.rosa_stage_ratio) * float(total_epochs)
            stage_start_step = int(math.ceil(stage_progress_epochs * float(steps_per_epoch)))
            stage_start_step = max(0, min(total_steps, stage_start_step))
            stage_ratio = float(config.rosa_stage_ratio)

        schedule_info = {
            "stage_ratio": stage_ratio,
            "stage_progress_epochs": float(stage_progress_epochs),
            "stage_start_step": int(stage_start_step),
            "mask_steps": int(config.rosa_mask_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "total_epochs": int(total_epochs),
            "total_steps": int(total_steps),
            "using_warmup_steps": int(using_warmup_steps),
        }
        self._rosa_stage_start_steps[adapter_name] = stage_start_step
        self._rosa_stage_schedule_meta[adapter_name] = schedule_info
        return schedule_info

    def get_stage_schedule(self, adapter_name: str = "default") -> dict[str, float | int]:
        if adapter_name not in self._rosa_stage_schedule_meta:
            raise RuntimeError("UniLoRA-RoSA-Stage training schedule is not initialized. Call set_training_schedule first.")
        return dict(self._rosa_stage_schedule_meta[adapter_name])

    def _get_stage_start_step(self, adapter_name: str = "default") -> int:
        if adapter_name not in self._rosa_stage_start_steps:
            raise RuntimeError("UniLoRA-RoSA-Stage training schedule is not initialized. Call set_training_schedule first.")
        return int(self._rosa_stage_start_steps[adapter_name])

    def should_collect_gradients(self, global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSAStageConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        stage_start_step = self._get_stage_start_step(adapter_name)
        return stage_start_step <= global_step < (stage_start_step + config.rosa_mask_steps)

    def should_generate_masks(self, next_global_step: int, adapter_name: str = "default") -> bool:
        config: UniLoRARoSAStageConfig = self.peft_config[adapter_name]
        if config.rosa_density <= 0.0 or config.rosa_mask_steps <= 0 or self.has_sparse_masks(adapter_name):
            return False
        stage_start_step = self._get_stage_start_step(adapter_name)
        return next_global_step >= (stage_start_step + config.rosa_mask_steps)


class UniLoRARoSAStageSnipModel(UniLoRARoSAStageModel):
    """
    Stage-wise RoSA whose sparse mask uses SNIP |W_ij * g_ij| saliency.
    """

    prefix: str = "unilora_rosa_stage_snip_"

    def set_training_schedule(
        self,
        total_steps: int,
        steps_per_epoch: int,
        total_epochs: int,
        stage_start_step_override: int | None = None,
        adapter_name: str = "default",
    ) -> dict[str, float | int]:
        schedule_info = super().set_training_schedule(
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            stage_start_step_override=stage_start_step_override,
            adapter_name=adapter_name,
        )
        schedule_info["score_mode"] = "snip"
        self._rosa_stage_schedule_meta[adapter_name] = schedule_info
        return schedule_info

    def accumulate_gradient_statistics(self, adapter_name: str = "default") -> dict[str, int]:
        updated_modules = 0
        updated_tensors = 0
        for module in self._iter_unilora_modules():
            updated = module.accumulate_snip_statistics(adapter_name)
            if updated > 0:
                updated_modules += 1
                updated_tensors += updated
        return {"updated_modules": updated_modules, "updated_tensors": updated_tensors}
