from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState

from .model import UniLoRASwapModel


class UniLoRASwapCallback(TrainerCallback):
    """
    Trainer callback that periodically triggers UniLoRA-Swap bucket reassignment.
    """

    def __init__(self, adapter_name: str = "default") -> None:
        self.adapter_name = adapter_name

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step <= 0:
            return control

        swap_model = self._resolve_swap_model(kwargs.get("model"))
        optimizer = kwargs.get("optimizer")
        if swap_model is None or optimizer is None:
            return control

        config = swap_model.peft_config[self.adapter_name]
        if config.swap_interval_steps <= 0:
            return control
        if state.global_step < config.swap_start_after_steps:
            return control
        if state.global_step % config.swap_interval_steps != 0:
            return control

        swap_model.perform_swap(optimizer=optimizer, adapter_name=self.adapter_name)
        return control

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch is None:
            return control

        swap_model = self._resolve_swap_model(kwargs.get("model"))
        optimizer = kwargs.get("optimizer")
        if swap_model is None or optimizer is None:
            return control

        config = swap_model.peft_config[self.adapter_name]
        if config.swap_interval_epochs <= 0:
            return control

        epoch = int(state.epoch)
        if epoch < config.swap_start_after_epochs:
            return control
        if epoch <= 0 or epoch % config.swap_interval_epochs != 0:
            return control

        swap_model.perform_swap(optimizer=optimizer, adapter_name=self.adapter_name)
        return control

    @staticmethod
    def _resolve_swap_model(model) -> UniLoRASwapModel | None:
        if isinstance(model, UniLoRASwapModel):
            return model
        base_model = getattr(model, "base_model", None)
        if isinstance(base_model, UniLoRASwapModel):
            return base_model
        return None
