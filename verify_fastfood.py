import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("NLU/peft/src"))

from peft import get_peft_model, UniLoRAFastFoodConfig, PeftType

def test_fastfood_minimal():
    print("Initializing minimal model...")
    # Create a simple linear model
    base_model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    
    # Configure UniLoRA-FastFood
    config = UniLoRAFastFoodConfig(
        r=2,
        theta_d_length=64,
        target_modules=[".*"], # Match all linear layers
        init_theta_d_bound=0.02,
    )
    
    print("Injecting UniLoRA-FastFood...")
    model = get_peft_model(base_model, config)
    
    # Check if theta_d exists
    theta_d_name = "base_model.unilora_fastfood_theta_d.default"
    if hasattr(model, "unilora_fastfood_theta_d"):
        print(f"Success: Found {theta_d_name}")
    else:
        # Check in named parameters
        found = any("unilora_fastfood_theta_d" in n for n, _ in model.named_parameters())
        if found:
             print(f"Success: Found unilora_fastfood_theta_d in parameters")
        else:
            raise RuntimeError("Failed to find shared theta_d parameter!")

    # Test forward
    print("Testing forward pass...")
    x = torch.randn(1, 10)
    output = model(x)
    print(f"Forward output shape: {output.shape}")
    
    # Test backward
    print("Testing backward pass...")
    loss = output.sum()
    loss.backward()
    
    # Check if theta_d has gradient
    theta_param = model.unilora_fastfood_theta_d["default"]
    if theta_param.grad is not None:
        grad_norm = theta_param.grad.norm().item()
        print(f"Success: theta_d gradient norm = {grad_norm:.6f}")
        if grad_norm > 0:
            print("Validation PASSED: Model is trainable with FastFood projection.")
        else:
            print("Warning: Gradient is zero. Check projection logic.")
    else:
        print("Error: theta_d has NO gradient!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_fastfood_minimal()
    except Exception as e:
        print(f"Validation FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
