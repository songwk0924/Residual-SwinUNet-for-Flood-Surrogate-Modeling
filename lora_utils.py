# -----------------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) Utilities
#
# Official implementation for the paper:
# "Enhancing Cross-Regional Transferability of Deep Learning-Based Flood Surrogate Models 
#  to Data-Scarce Catchments"
#
# Authors: Wenke Song, Mingfu Guan
# Affiliation: Department of Civil Engineering, The University of Hong Kong
# Contact: songwk@connect.hku.hk
#
# Method Reference:
# Hu, E.J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models. DOI:10.48550/arxiv.2106.09685
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import math
import types
from typing import List, Dict, Optional, Union

class LoRALinear(nn.Module):
    """
    Implementation of LoRA (Low-Rank Adaptation) for Linear layers.
    
    This module wraps an existing Linear layer, freezes its weights, and adds 
    a parallel low-rank branch (A -> B) that is trainable.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, alpha: int = 16, dropout_p: float = 0.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Save original weights (frozen)
        self.weight = linear_layer.weight.detach().requires_grad_(False)
        self.bias = None
        if hasattr(linear_layer, 'bias') and linear_layer.bias is not None:
            self.bias = linear_layer.bias.detach().requires_grad_(False)
        
        # Low-rank adaptation matrices: A (rank, in) and B (out, rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output
        base_output = nn.functional.linear(x, self.weight, self.bias)
        
        # LoRA path: x -> A -> dropout -> B
        lora_output = (self.lora_B @ self.dropout(self.lora_A @ x.transpose(-1, -2))).transpose(-1, -2)
        
        return base_output + self.scaling * lora_output

def apply_lora_to_model(
    model: nn.Module, 
    target_modules: Optional[List[str]] = None, 
    rank: int = 16, 
    alpha: int = 16, 
    dropout: float = 0.0, 
    layer_config: Optional[Dict[str, Dict]] = None
) -> nn.Module:
    """
    Apply LoRA to specific layers of a PyTorch model.
    
    Args:
        model (nn.Module): The backbone model to be adapted.
        target_modules (List[str], optional): List of module names to apply LoRA to. 
            Defaults to ["qkv", "proj", "fc1", "fc2", "conv"].
        rank (int): Default LoRA rank. Defaults to 16.
        alpha (int): Default LoRA scaling factor. Defaults to 16.
        dropout (float): Dropout probability for LoRA layers. Defaults to 0.0.
        layer_config (Dict, optional): Specific configuration for certain layer types.
            Example: {"conv": {"rank": 8, "alpha": 16}}.
    
    Returns:
        nn.Module: The modified model with LoRA layers injected.
    """
    if layer_config is None:
        layer_config = {}
    
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2", "conv"]
    
    # Freeze all original model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    lora_count = 0
    
    # Iterate through all modules to find targets
    for name, module in model.named_modules():
        # Check if current module matches any target string
        is_target = any(target in name for target in target_modules)
        
        if is_target:
            # Determine rank, alpha, and dropout for this specific layer
            current_rank = rank
            current_alpha = alpha
            current_dropout = dropout
            
            # Apply specific configs if defined in layer_config
            for layer_type, config in layer_config.items():
                if layer_type in name:
                    current_rank = config.get("rank", rank)
                    current_alpha = config.get("alpha", alpha)
                    current_dropout = config.get("dropout", dropout)
                    break
            
            # Replace nn.Linear with LoRALinear
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                # Retrieve parent module to set attribute
                parent_module = model
                if parent_name:
                    for part in parent_name.split('.'):
                        if part.isdigit():
                            parent_module = parent_module[int(part)]
                        else:
                            parent_module = getattr(parent_module, part)
                
                # Replace with LoRALinear wrapper
                setattr(parent_module, child_name, 
                       LoRALinear(module, rank=current_rank, alpha=current_alpha, dropout_p=current_dropout))
                lora_count += 1
                print(f"Applying LoRA to Linear layer: {name} (rank={current_rank}, alpha={current_alpha}, dropout={current_dropout})")
            
    # Enable gradients ONLY for LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nSuccessfully applied LoRA to {lora_count} layers.")
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)\n")
    
    return model

def save_lora_weights(model: nn.Module, path: str):
    """
    Save only the trainable LoRA weights (lora_A and lora_B) to reduce storage.
    
    Args:
        model (nn.Module): The model with LoRA layers.
        path (str): File path to save the weights.
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param
    
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to: {path}")

def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load LoRA weights into a model.
    
    Args:
        model (nn.Module): The initialized model (already applied with loRA structure).
        path (str): Path to the saved LoRA weights.
        
    Returns:
        nn.Module: Model with loaded weights.
    """
    lora_state_dict = torch.load(path, map_location='cpu') # Added map_location for safety
    
    # Load weights into the model state dict
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
        else:
            print(f"Warning: Key {name} not found in model state dict.")
    
    model.load_state_dict(model_state_dict)
    print(f"Successfully loaded LoRA weights from {path}")
    
    return model