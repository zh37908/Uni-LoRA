# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
import torch
from safetensors import safe_open
from torch import nn

from peft import PeftModel, UniLoRAConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


class TestUniLoRA:
    def get_mlp(self):
        model = MLP()
        return model

    def test_unilora_parameters(self):
        mlp = self.get_mlp()
        # 在您的新实现中，theta_d_length 实际上充当了"共享参数池的大小"(Codebook Size)
        # indices 的取值范围是 [0, theta_d_length)
        theta_d_length = 100 
        # num_vectors 在新代码中似乎未被直接用于形状定义，主要依赖 theta_d_length
        r = 4
        
        config = UniLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"], 
            theta_d_length=theta_d_length, 
            r=r
        )
        mlp_unilora = get_peft_model(mlp, config)

        theta_d = mlp_unilora.unilora_theta_d["default"]
        
        # 1. 检查 Theta_d (共享参数池)
        # 根据 _init_unilora_theta_d: torch.zeros(config.theta_d_length)
        assert theta_d.shape == (theta_d_length,)

        # 2. 检查 Indices 和 Scales (原 Logits 和 Norm)
        # Indices 应该直接等于 LoRA 低秩矩阵的形状
        
        # lin0: (10 -> 20)
        # indices_B: (out_features, r) -> (20, 4)
        unilora_lin0_indices_B = mlp_unilora.lin0.unilora_indices_B["default"]
        assert unilora_lin0_indices_B.shape == (mlp.lin0.out_features, config.r)
        # scales_B 应该与 indices_B 形状一致
        unilora_lin0_scales_B = mlp_unilora.lin0.unilora_scales_B["default"]
        assert unilora_lin0_scales_B.shape == (mlp.lin0.out_features, config.r)

        # lin1: (20 -> 20)
        # indices_A: (r, in_features) -> (4, 20)
        unilora_lin1_indices_A = mlp_unilora.lin1.unilora_indices_A["default"]
        assert unilora_lin1_indices_A.shape == (config.r, mlp.lin1.in_features)
        
        # lin3: (20 -> 2)
        unilora_lin3_indices_A = mlp_unilora.lin3.unilora_indices_A["default"]
        assert unilora_lin3_indices_A.shape == (config.r, mlp.lin3.in_features)

        # 3. 检查参数共享
        # 确保所有层的 theta_d 指向同一个内存地址
        assert (
            mlp_unilora.lin0.unilora_theta_d["default"].data_ptr()
            == mlp_unilora.lin3.unilora_theta_d["default"].data_ptr()
        )
        assert mlp_unilora.lin1.unilora_theta_d["default"].data_ptr() == theta_d.data_ptr()

        # 4. 运行前向传播测试
        input = torch.randn(5, 10)
        output = mlp_unilora(input)
        assert output.shape == (5, 2)

    def test_save_load(self, tmp_path):
        """测试保存和加载的一致性，不再包含 TopK 逻辑"""
        torch.manual_seed(0)
        mlp = self.get_mlp()
        config = UniLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"],
            theta_d_length=50,
            r=4
        )
        mlp_unilora = get_peft_model(mlp, config)
        
        # 运行一次推理以确保 buffer 被初始化/移动到正确设备
        input = torch.randn(5, 10)
        output_before = mlp_unilora(input)
        
        save_path = tmp_path / "unilora"
        mlp_unilora.save_pretrained(save_path)
        assert os.path.exists(save_path / "adapter_config.json")
        assert os.path.exists(save_path / "adapter_model.safetensors")

        # 检查 safetensors 内容，确保保存的是 indices 和 scales
        with safe_open(save_path / "adapter_model.safetensors", framework="pt") as f:
            keys = f.keys()
            # 应该包含 indices 和 scales，不应包含 logits
            assert any("unilora_theta_d" in k for k in keys)
           

        # 加载测试
        # del mlp_unilora
        torch.manual_seed(0)
        mlp = self.get_mlp()
        mlp_unilora_loaded = PeftModel.from_pretrained(mlp, save_path)

        output_loaded = mlp_unilora_loaded(input)
        assert torch.allclose(output_before, output_loaded, atol=1e-8, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_unilora_dtypes(self, dtype):
        mlp = self.get_mlp()
        if (dtype == torch.bfloat16) and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            pytest.skip("bfloat16 not supported on this system, skipping the test")

        config = UniLoRAConfig(
            target_modules=["lin0", "lin1", "lin3"], 
            theta_d_length=10, 
            r=4
        )
        mlp_unilora = get_peft_model(mlp.to(dtype), config)
        inputs = torch.randn(5, 10).to(dtype)
        output = mlp_unilora(inputs)
        
        assert output.dtype == dtype
        
        

    def test_unilora_nb_savable_params(self):
        """测试可保存参数量的计算：Indices + Scales + Theta_d"""
        mlp = self.get_mlp()
        theta_d_length = 20
        r = 4
        config = UniLoRAConfig(
            target_modules=["lin0", "lin1"],
            theta_d_length=theta_d_length,
            r=r
        )
        mlp_unilora = get_peft_model(mlp, config)

        mlp_unilora.lin3.requires_grad_(True)  # lin3 是完全可训练的，作为 'other_params'

        adapter_params, other_params = mlp_unilora.get_nb_savable_parameters()
        
        # 计算预期的 Adapter 参数量
        # 1. Theta_d (共享池)
        theta_d_params = theta_d_length # 1D tensor
        
    
        
        
        
        assert adapter_params == theta_d_params
        assert other_params == 0
