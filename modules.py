import math
from typing import Dict, Tuple, List

import torch
from transformers import BertModel, ViTModel
from transformers.tokenization_utils_base import BatchEncoding

def get_global_mask(mask_raw):
    global_mask = torch.ones(mask_raw[:, :1].shape, device=mask_raw.device)
    return torch.cat([global_mask, mask_raw], dim=1)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = [batch, seq_len, d_model]
        """
        return self.pe.permute(1,0,2)[:, :x.size(1), :]


class TemporalBlockEncodingLayer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 nhead: int,
                 dropout: float,
                 activation: str) -> None:
        super().__init__()

        # Intra-Temporal Attention
        self.intra_norm = torch.nn.LayerNorm(d_model)
        self.intra_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.intra_dropout = torch.nn.Dropout(dropout)
        
        # Inter-Temporal Attention
        self.inter_norm = torch.nn.LayerNorm(d_model)
        self.inter_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.inter_dropout = torch.nn.Dropout(dropout)

        # Feed Forward
        if activation == "gelu": self.activation = torch.nn.GELU()
        elif activation == "relu": self.activation = torch.nn.ReLU()
        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff)
        self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout)
        self.ff_dropout2 = torch.nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        x = src
        # Intra-Temporal Attention
        intra_output, intra_weight = self.intra_block(self.intra_norm(x))
        x = x + intra_output

        # Inter-Temporal Attention
        inter_output, inter_weight = self.inter_block(self.inter_norm(x), observed_mask)
        x = x + inter_output

        # Feed Forward
        ff_output = self.ff_block(self.ff_norm(x))
        x = x + ff_output

        return x
    
    def intra_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_modality, d_model = x.shape
        x = x.view(-1, num_modality, d_model)
        
        output, weight = self.intra_attn(x, x, x)
        output = output.view(batch_size, seq_len, num_modality, d_model)
        weight = weight.view(batch_size, -1, num_modality, num_modality)
        return self.intra_dropout(output), weight
    
    def inter_block(self, x: torch.Tensor, mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, num_modality, d_model]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, num_modality, d_model = x.shape
        x = x.permute(0,2,1,3).reshape(-1, seq_len, d_model)
        mask = mask.unsqueeze(-2).expand(-1, num_modality, -1).reshape(-1, seq_len)

        output, weight = self.inter_attn(x, x, x, key_padding_mask=mask)
        output = output.view(batch_size, num_modality, seq_len, d_model).permute(0,2,1,3)
        return self.inter_dropout(output), weight
        
    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        output = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(output)

class TemporalBlockEncoder(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 nhead: int,
                 num_layers: int,
                 dropout: float,
                 activation: str,
                 temporal_cols: List[str]):
        super().__init__()

        self.temporal_cols = temporal_cols
        self.block_encoding_layers = torch.nn.ModuleList([TemporalBlockEncodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])

    def forward(self, observed_dict, observed_mask):
        src = torch.stack([observed_dict[col] for col in self.temporal_cols], dim=-2)
        observed_mask = torch.where(get_global_mask(observed_mask)==1, 0, -torch.inf)
        for mod in self.block_encoding_layers:
            src = mod(src, observed_mask)
        
        encoding = src[:, :, 0, :] # Obtain only the global temporal token
        
        return encoding, observed_mask


class MetadataEncoder(torch.nn.Module):
    """
    [CLS] token is already included
    """
    def __init__(self, d_model, nhead, d_ff, dropout, activation, num_layers):
        super().__init__()
        self.transformer_encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation=activation, batch_first=True, norm_first=True), num_layers)
    
    def forward(self, metadata_encoding, mask):
        mask = torch.where(mask==1, 0, -torch.inf)
        encoding = self.transformer_encoder(metadata_encoding, src_key_padding_mask=mask)
        
        return encoding


class TextEncoder(torch.nn.Module):
    def __init__(self, resize_token_len=None):
        super().__init__()
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        if resize_token_len: self.text_model.resize_token_embeddings(resize_token_len)
    
    def forward(self, text_input: BatchEncoding) -> torch.Tensor:
        return self.text_model(**text_input).last_hidden_state


class ImgEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    
    def forward(self, img_input: torch.Tensor) -> torch.Tensor:
        return self.img_model(img_input).last_hidden_state


class TemporalBlockDecodingLayer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 nhead: int,
                 dropout: float,
                 activation: str,
                 ) -> None:
        super().__init__()

        # Intra-Temporal Attention
        self.intra_norm = torch.nn.LayerNorm(d_model)
        self.intra_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.intra_dropout = torch.nn.Dropout(dropout)

        # Inter-Temporal Attention
        self.inter_norm = torch.nn.LayerNorm(d_model)
        self.inter_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.inter_dropout = torch.nn.Dropout(dropout)

        # Fusion Attention
        self.fusion_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.fusion_norm = torch.nn.LayerNorm(d_model)
        self.fusion_dropout = torch.nn.Dropout(dropout)

        # Feed Forward
        if activation == "gelu": self.activation = torch.nn.GELU()
        elif activation == "relu": self.activation = torch.nn.ReLU()
        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff_linear1 = torch.nn.Linear(d_model, d_ff)
        self.ff_linear2 = torch.nn.Linear(d_ff, d_model)
        self.ff_dropout1 = torch.nn.Dropout(dropout)
        self.ff_dropout2 = torch.nn.Dropout(dropout)

    
    def forward(self, tgt, multimodal, known_mask, multimodal_mask):
        x = tgt

        # Intra-Temporal Attention
        intra_output, intra_weight = self.intra_block(self.intra_norm(x))
        x = x + intra_output

        # Inter-Temporal Attention
        inter_output, inter_weight = self.inter_block(self.inter_norm(x), known_mask)
        x = x + inter_output

        # Fusion
        fusion_output = self.fusion_block(self.fusion_norm(x), multimodal, multimodal_mask)
        x = x + fusion_output

        # Feed Forward
        ff_output = self.ff_block(self.ff_norm(x))
        x = x + ff_output

        return x

    def intra_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_modality, d_model = x.shape
        x = x.view(-1, num_modality, d_model)
        
        output, weight = self.intra_attn(x, x, x)
        output = output.view(batch_size, seq_len, num_modality, d_model)
        weight = weight.view(batch_size, -1, num_modality, num_modality)
        return self.intra_dropout(output), weight

    def inter_block(self, x: torch.Tensor, mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, num_modality, d_model]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, num_modality, d_model = x.shape
        x = x.permute(0,2,1,3).reshape(-1, seq_len, d_model)
        mask = mask.unsqueeze(-2).expand(-1, num_modality, -1).reshape(-1, seq_len)

        output, weight = self.inter_attn(x, x, x, key_padding_mask=mask)
        output = output.view(batch_size, num_modality, seq_len, d_model).permute(0,2,1,3)
        return self.inter_dropout(output), weight

    def fusion_block(self, tgt, multimodal, multimodal_mask):
        # Prepare tgt
        batch_size, tgt_seq_len, num_modality, d_model = tgt.shape
        tgt = tgt.permute(0,2,1,3).reshape(-1, tgt_seq_len, d_model)

        # Prepare multimodal
        batch_size, multimodal_seq_len, d_model = multimodal.shape
        multimodal = multimodal.unsqueeze(1).expand(-1, num_modality, -1, -1).reshape(-1, multimodal_seq_len, d_model)
        multimodal_mask = multimodal_mask.unsqueeze(1).expand(-1, num_modality, -1).reshape(-1, multimodal_seq_len)

        # Attention
        output, weight = self.fusion_attn(tgt, multimodal, multimodal, key_padding_mask=multimodal_mask)
        output = output.view(batch_size, num_modality, tgt_seq_len, d_model).permute(0,2,1,3)
        return self.fusion_dropout(output)

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        output = self.ff_linear2(self.ff_dropout1(self.activation(self.ff_linear1(x))))
        return self.ff_dropout2(output)

class TemporalBlockDecoder(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 nhead: int,
                 dropout: float,
                 activation: str,
                 num_layers: int,
                 temporal_cols: List[str],
                 ) -> None:
        super().__init__()
        self.temporal_cols = temporal_cols
        self.modality_embedding = torch.nn.Embedding(4, d_model)
        self.multimodal_norm = torch.nn.LayerNorm(d_model)
        self.block_decoding_layers = torch.nn.ModuleList([TemporalBlockDecodingLayer(d_model, d_ff, nhead, dropout, activation) for _ in range(num_layers)])
    
    def forward(self,
                known_embedding_dict: Dict[str, torch.Tensor],
                observed_embedding: torch.Tensor,
                metadata_embedding: torch.Tensor,
                text_embedding: torch.Tensor,
                img_embedding: torch.Tensor,
                
                known_mask: torch.Tensor,
                observed_mask: torch.Tensor,
                metadata_mask: torch.Tensor,
                text_mask: torch.Tensor,
                img_mask: torch.Tensor,
                ) -> torch.Tensor:

        # Prepare Temporal Block Attention
        tgt_li = []
        for col in ["global_known"] + self.temporal_cols:
            tgt_li.append(known_embedding_dict[col])
        tgt = torch.stack(tgt_li, dim=-2)
        known_mask = torch.where(get_global_mask(known_mask)==1, 0, -torch.inf)

        # Prepare Fusion
        multimodal_li, mod_emb_li = [], []
        for modailty_idx, multimodal in enumerate([observed_embedding, metadata_embedding, text_embedding, img_embedding]):
            modality = self.modality_embedding(torch.full(multimodal.shape[:-1], modailty_idx, device=multimodal.device))
            multimodal_li.append(multimodal)
            mod_emb_li.append(modality)
        
        multimodal = self.multimodal_norm(torch.cat(multimodal_li, dim=1))
        multimodal = multimodal + torch.cat(mod_emb_li, dim=1)
        multimodal_mask = torch.cat([observed_mask, metadata_mask, text_mask, img_mask], dim=1)
        multimodal_mask = torch.where(multimodal_mask==1, 0, -torch.inf)
        
        # Temporal Block Decoding
        for mod in self.block_decoding_layers:
            tgt = mod(tgt, multimodal, known_mask, multimodal_mask)
        
        decoding = tgt[:, :, 0, :]
        return decoding

