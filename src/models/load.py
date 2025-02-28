import torch
from typing import Any, Dict, Optional
from .outfit_transformer import (
    OutfitTransformerConfig, 
    OutfitTransformer
)
from .outfit_clip_transformer import (
    OutfitCLIPTransformerConfig,
    OutfitCLIPTransformer
)


def load_model(model_type, checkpoint, **cfg_kwargs):
    if checkpoint:
        state_dict = torch.load(checkpoint)

    if model_type == 'original':
        cfg = OutfitTransformerConfig(**state_dict['config']) if checkpoint else OutfitTransformerConfig(**cfg_kwargs)
        model = OutfitTransformer(cfg)

    elif model_type == 'clip':
        cfg = OutfitCLIPTransformerConfig(**state_dict['config']) if checkpoint else OutfitCLIPTransformerConfig(**cfg_kwargs)
        model = OutfitCLIPTransformer(cfg)
        
    if checkpoint:
        new_state_dict = {}
        for key, value in state_dict['model'].items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value
        
        model.load_state_dict(
            new_state_dict, strict=False
        )
        print(
            f"Loaded model from checkpoint: {checkpoint}"
        )
            
    return model