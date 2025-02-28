from torch import nn
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from torch import Tensor
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import pathlib
from ..data.datatypes import (
    FashionCompatibilityQuery, FashionComplementaryQuery, FashionItem
)
from .modules.encoder import ItemEncoder
from ..utils.model_utils import get_device

@dataclass
class OutfitTransformerConfig:
    padding: Literal['longest', 'max_length'] = 'longest'
    max_length: int = 16
    truncation: bool = True
    
    query_img_path = pathlib.Path(__file__).parent.absolute() / 'question.jpg'
    
    init_item_enc: bool = True
    item_enc_text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    item_enc_dim_per_modality: int = 128
    item_enc_norm_out: bool = True
    aggregation_method: Literal['concat', 'sum', 'mean', 'concat_all', 'concat-3'] = 'concat-3'
    
    init_transformer: bool = True
    transformer_n_head: int = 16
    transformer_d_ffn: int = 2024
    transformer_n_layers: int = 6
    transformer_dropout: float = 0.3
    transformer_norm_out: bool = False
    
    d_embed: int = 128


class OutfitTransformer(nn.Module):
    
    def __init__(
        self, 
        cfg: Optional[OutfitTransformerConfig] = None
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else OutfitTransformerConfig()
        
        # Initialize the item encoder
        self._build_item_enc()
        image_size = (self.item_enc.image_size, self.item_enc.image_size)
        self.image_query = cv2.resize(
            src=cv2.cvtColor(cv2.imread(str(cfg.query_img_path)), 
                            cv2.COLOR_BGR2RGB), 
            dsize=image_size
        )
        self.image_pad = np.array(
            Image.new("RGB", image_size)
        )
        self.text_pad = ''
            
        # Initialize the style encoder
        self._build_style_enc()
        self._build_predict_ffn()
        self._init_predict_s_emb()
        self._build_embed_ffn()
            
        
    def _build_item_enc(self) -> ItemEncoder:
        """Builds the outfit encoder using configuration parameters."""
        self.item_enc = ItemEncoder(
            model_name=self.cfg.item_enc_text_model_name,
            enc_dim_per_modality=self.cfg.item_enc_dim_per_modality,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
    
    def _build_style_enc(self) -> nn.TransformerEncoder:
        """Builds the transformer encoder using configuration parameters."""
        style_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.item_enc.d_embed,
            nhead=self.cfg.transformer_n_head,
            dim_feedforward=self.cfg.transformer_d_ffn,
            dropout=self.cfg.transformer_dropout,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        style_enc_norm = nn.LayerNorm(
            self.item_enc.d_embed
        )
        self.style_enc = nn.TransformerEncoder(
            encoder_layer=style_enc_layer, 
            num_layers=self.cfg.transformer_n_layers,
            norm=style_enc_norm,
            enable_nested_tensor=False
        )
    
    def _build_predict_ffn(self) -> nn.Sequential:
        """Builds the feed-forward classifier layer."""
        self.predict_ffn = nn.Sequential(
            nn.LayerNorm(self.item_enc.d_embed),
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.item_enc.d_embed, 1, bias=False),
            nn.Sigmoid()
        )
        
    def _build_embed_ffn(self) -> nn.Sequential:
        """Builds the feed-forward embedding layer."""
        self.embed_ffn = nn.Sequential(
            nn.Linear(self.item_enc.d_embed, self.cfg.d_embed, bias=False)
        )
        
    def _init_predict_s_emb(self):
        """Initializes the classifier embedding."""
        self.predict_s_emb = nn.Parameter(
            torch.randn(1, self.item_enc.d_embed) * 0.02, 
            requires_grad=True
        )
    
    def _get_max_length(self, sequences):
        """Determine max length based on configuration."""
        if self.cfg.padding == 'max_length':
            return self.cfg.max_length
        max_length = max(len(seq) for seq in sequences)
        return min(self.cfg.max_length, max_length) if self.cfg.truncation else max_length

    def _pad_sequences(self, sequences, pad_value, max_length):
        """Pad sequences to the specified max length."""
        return [seq[:max_length] + [pad_value] * (max_length - len(seq)) for seq in sequences]

    def _pad_and_mask_outfits(
        self, outfits: List[List[FashionItem]]
    ) -> Tuple[List[List[np.ndarray]], List[List[str]], torch.Tensor]:
        max_length = self._get_max_length(outfits)
        
        images = self._pad_sequences(
            [[item.image for item in outfit] for outfit in outfits], 
            self.image_pad, max_length
        )
        texts = self._pad_sequences(
            [[f"A photo of a {item.description}".lower() for item in outfit] for outfit in outfits], 
            self.text_pad, max_length
        )
        mask = [[0] * len(seq) + [1] * (max_length - len(seq)) for seq in outfits]
        
        return images, texts, torch.BoolTensor(mask).to(self.device)

    def _pad_and_mask_for_embs(
        self, embs_of_outfits: List[List[np.ndarray]]
    ) -> Tuple[Tensor, Tensor]:
        max_length = self._get_max_length(embs_of_outfits)
        batch_size = len(embs_of_outfits)
        
        embeddings = torch.zeros((batch_size, max_length, self.item_enc.d_embed), 
                                 dtype=torch.float, device=self.device)
        mask = torch.ones((batch_size, max_length), 
                          dtype=torch.bool, device=self.device)
        for i, embs_of_outfit in enumerate(embs_of_outfits):
            embs_of_outfit = np.array(embs_of_outfit[:max_length])
            length = len(embs_of_outfit)
            embeddings[i, :length] = torch.as_tensor(embs_of_outfit, dtype=torch.float32, device=self.device)
            mask[i, :length] = False
        
        return embeddings, mask
    
    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return get_device(self)
    
    def forward(
        self, 
        inputs: Union[
            List[FashionCompatibilityQuery], 
            List[FashionComplementaryQuery],
            List[FashionItem],
        ], 
        *args, **kwargs
    ) -> Tensor:
        if isinstance(inputs[0], FashionCompatibilityQuery):
            return self.predict_score(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionComplementaryQuery):
            return self.embed_complementary_query(inputs, *args, **kwargs)
        
        elif isinstance(inputs[0], FashionItem):
            return self.embed_complementary_item(inputs, *args, **kwargs)
        else:
            raise ValueError("Invalid input type.")
    
    def predict_score(
        self, 
        query: List[FashionCompatibilityQuery],
        use_precomputed_embedding: bool = False,
        *args, **kwargs
    ) -> Tensor:
        outfits = [query_.outfit for query_ in query]
        
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            embs_of_outfits = [[item_.embedding for item_ in outfit] for outfit in outfits]
            enc_outs, mask = self._pad_and_mask_for_embs(embs_of_outfits)
        else:
            outfits = [query_.outfit for query_ in query]
            images, texts, mask = self._pad_and_mask_outfits(outfits)
            enc_outs = self.item_enc(images, texts)

        predict_s_emb = torch.nn.functional.normalize(self.predict_s_emb, p=2, dim=-1)
        predict_s_emb = predict_s_emb.unsqueeze(0).expand(len(query), -1, -1)
        enc_outs = torch.cat([predict_s_emb, enc_outs], dim=1)

        mask = torch.cat([torch.zeros(len(query), 1, dtype=torch.bool, device=self.device), mask], dim=1)

        last_hidden_states = self.style_enc(enc_outs, src_key_padding_mask=mask)
        scores = self.predict_ffn(last_hidden_states[:, 0, :])

        return scores
    
    def embed_complementary_query(
        self, 
        query: List[FashionComplementaryQuery], 
        use_precomputed_embedding: bool=False,
        *args, **kwargs
    ) -> Tensor:
        """
        Embeds query items for compatibility.
        """
        q_items = [[FashionItem(category=i.category, image=self.image_query, description=i.category)] for i in query]
        outfits = [query_.outfit for query_ in query]
        
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in sum(outfits, [])])
            
            q_images, q_texts, q_mask = self._pad_and_mask_outfits(q_items)
            q_enc_outs = self.item_enc(q_images, q_texts)
            
            embs_of_outfits = [[item_.embedding for item_ in outfit] for outfit in outfits]
            enc_outs, mask = self._pad_and_mask_for_embs(embs_of_outfits)
            
            enc_outs = torch.cat([q_enc_outs, enc_outs], dim=1)
            mask = torch.cat([q_mask, mask], dim=1)
        else:
            outfits = [q_item + outfit for q_item, outfit in zip(q_items, outfits)]
            images, texts, mask = self._pad_and_mask_outfits(outfits)
            enc_outs = self.item_enc(images, texts)
        
        last_hidden_states = self.style_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :])
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings # [B, D]

    def embed_complementary_item(
        self, 
        item: List[FashionItem], 
        use_precomputed_embedding: bool=False,
        *args, **kwargs
    ) -> Tensor:
        """
        Embeds candidate items for compatibility.
        """
        if use_precomputed_embedding:
            assert all([item_.embedding is not None for item_ in item])
            embs_of_outfits = [[item_.embedding] for item_ in item]
            enc_outs, mask = self._pad_and_mask_for_embs(embs_of_outfits)
        else:
            outfits = [[item_] for item_ in item]
            images, texts, mask = self._pad_and_mask_outfits(outfits)
            enc_outs = self.item_enc(images, texts)
        
        last_hidden_states = self.style_enc(enc_outs, src_key_padding_mask=mask)
        embeddings = self.embed_ffn(last_hidden_states[:, 0, :]) # [B, D]
        
        if self.cfg.transformer_norm_out:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings # [B, D]