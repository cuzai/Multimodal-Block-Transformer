
import torch
from typing import Dict, Tuple, List

from modules import PositionalEncoding, TemporalBlockEncoder, MetadataEncoder, TextEncoder, ImgEncoder, TemporalBlockDecoder

class MultimodalBlockTransformer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 d_proj: int,
                 nhead: int,
                 num_layers: int,
                 dropout: float,
                 activation: str,
                 temporal_catcol_dict: Dict[str, int],
                 temporal_numcol_dict: Dict[str, int],
                 temporal_target_col: str,
                 metadata_col_dict: Dict[str, int],
                 text_resize_token_len: int = None,
                 ) -> None:
        """
        Args:
            d_model (int): Hidden size of the model.
            d_ff (int): Hidden size of the feed-forward layer.
            d_proj (int): Hidden size of the projection layer.
            nhead (int): Number of heads in the multiheadattention models.
            num_layers (int): Number of sub-encoder-layers in the encoder.
            dropout (float): Dropout value.
            activation (str): The activation function of intermediate layer, relu or gelu.
            temporal_catcol_dict (Dict[str, int]): Number of unique values for each categorical column in temporal data.
            temporal_numcol_dict (Dict[str, int]): Normalization statistics for each numerical column in temporal data.
            temporal_target_col (str): The target column in temporal data.
            metadata_col_dict (Dict[str, int]): Number of unique values for each categorical column in metadata.
            text_resize_token_len (int, optional): The length to resize text tokens to. Defaults to None.
        """
        super().__init__()
        self.temporal_catcol_dict, self.temporal_numcol_dict = temporal_catcol_dict, temporal_numcol_dict
        self.temporal_cols = list(self.temporal_catcol_dict.keys()) + list(self.temporal_numcol_dict.keys())
        
        # 1. Temporal Embedding
        self.global_observed_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_embedding_dict = torch.nn.ModuleDict()
        for col, num_cls in temporal_catcol_dict.items():
            self.temporal_embedding_dict[col] = torch.nn.Embedding(num_cls, d_model)
        for col in temporal_numcol_dict:
            self.temporal_embedding_dict[col] = torch.nn.Linear(1, d_model)
        ## 1.1 Positional Encoding, Modality Embedding
        self.pos_enc = PositionalEncoding(d_model)
        self.mod_emb = torch.nn.Embedding(len(self.temporal_cols)+1, d_model) # Reserve 0 for global temporal token

        # 2. Temporal Block Encoding
        self.temporal_block_encoder = TemporalBlockEncoder(d_model, d_ff, nhead, num_layers, dropout, activation, self.temporal_cols)
        self.observed_proj = torch.nn.Linear(d_model, d_proj)

        # 3. Attribute Encoding
        ## 3.1 Attribute Embedding
        self.metadata_catcol_embedding = torch.nn.Embedding(sum(metadata_col_dict.values()), d_model) # Values for numerical columns will be replaced by linear layer.
        self.metadata_numcol_linear = torch.nn.Linear(1, d_model)
        ## 3.2 Positional Encoding, Modality Embedding
        self.pos_enc = PositionalEncoding(d_model)
        self.mod_emb = torch.nn.Embedding(len(metadata_col_dict.keys())+1, d_model) # Reserve 0 for [CLS] token. [CLS] token is considered as a type of attribute.
        ## 3.3 Attribute Encoding
        self.metadata_encoder = MetadataEncoder(d_model, nhead, d_ff, dropout, activation, num_layers)
        self.metadata_proj = torch.nn.Linear(d_model, d_proj)
        
        # 4. Text Encoding
        self.text_encoder = TextEncoder(resize_token_len=text_resize_token_len)
        self.text_proj = torch.nn.Linear(768, d_proj) # 768 is the hidden size of the BERT model

        # 5. Image Encoding
        self.img_encoder = ImgEncoder()
        self.img_proj = torch.nn.Linear(768, d_proj) # 768 is the hidden size of the BERT model

        # 6. Known Embedding
        self.global_known_token = torch.nn.Parameter(torch.randn(1, 1, d_model))

        # 7. Temporal Block Decoding
        self.temporal_block_decoder = TemporalBlockDecoder(d_model, d_ff, nhead, dropout, activation, num_layers, [col for col in self.temporal_cols if col!=temporal_target_col])

        # Loss
        self.output = torch.nn.Linear(d_model, 1)
        self.mse_loss = torch.nn.MSELoss(reduction="none")
    
    def forward(self,
                observed_input: Dict[str, torch.Tensor],
                metadata_input: Dict[str, torch.Tensor],
                text_input: Dict[str, torch.Tensor],
                img_input: torch.Tensor,
                known_input: Dict[str, torch.Tensor],
                observed_mask: torch.Tensor,
                known_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            observed_input: dictionary of observed input features.
            observed_mask: mask for observed input features.
            metadata_input: dictionary of attribute input features.
                             It inclues categorical attributes and numerical attributes.
            text_input: dictionary of text input features.
            img_input: image input features.
            known_input: dictionary of known input features.
        """
        
        # 1. Observed Embedding
        observed_embedding_dict = {}
        batch_size, seq_len, _ = observed_input[list(observed_input.keys())[0]].shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        observed_embedding_dict["global_observed"] = self.global_observed_token.expand(batch_size, seq_len, -1)
        for col in self.temporal_cols:
            observed_embedding_dict[col] = self.temporal_embedding_dict[col](observed_input[col])
        ## 1.1 Positional Encoding, Modality Embedding
        for modality_idx, col in enumerate(["global_observed"] + self.temporal_cols):
            embedding = torch.cat([cls_token, observed_embedding_dict[col]], dim=1)
            pos_enc = self.pos_enc(embedding)
            mod_emb = self.mod_emb(torch.full(embedding.shape[:-1], modality_idx, device=embedding.device))
            observed_embedding_dict[col] = embedding + pos_enc + mod_emb
        
        # 2. Temporal Block Encoding
        observed_encoding, observed_mask = self.temporal_block_encoder(observed_embedding_dict, observed_mask)
        observed_encoding = self.observed_proj(observed_encoding)

        # 3. Metadata Encoding
        ## 3.1 Metadata Embedding
        input_ids = metadata_input["input_ids"]
        metadata_catcol_embedding = self.metadata_catcol_embedding(input_ids)
        metadata_numcol_position = metadata_input["num_position"].unsqueeze(-1).expand(-1, -1, metadata_catcol_embedding.shape[-1]) # Find the position of numerical attributes within `input_ids`
        metadata_numcol_embedding = self.metadata_numcol_linear(metadata_input["num_li"].unsqueeze(-2))
        metadata_embedding = torch.scatter(input=metadata_catcol_embedding, index=metadata_numcol_position, src=metadata_numcol_embedding, dim=1)
        ## 3.2 Positional Encoding, Modality Embedding
        pos_enc = self.pos_enc(metadata_embedding)
        mod_emb = self.mod_emb(metadata_input["token_type_ids"])
        metadata_embedding = metadata_embedding + pos_enc + mod_emb
        ### 3.3 Metadata Encoding
        metadata_encoding = self.metadata_encoder(metadata_embedding, metadata_input["attention_mask"])
        metadata_encoding = self.metadata_proj(metadata_encoding)

        # 4. Text Encoding
        text_encoding = self.text_encoder(text_input)
        text_encoding = self.text_proj(text_encoding)

        # 5. Image Encoding
        img_encoding = self.img_encoder(img_input)
        img_encoding = self.img_proj(img_encoding)
        img_mask = torch.ones(img_encoding.shape[:-1], device=img_encoding.device)

        # 6. Known Embedding
        known_embedding_dict = {}
        batch_size, seq_len, _ = known_input[list(known_input.keys())[0]].shape
        known_embedding_dict["global_known"] = self.global_known_token.expand(batch_size, seq_len, -1)
        for col in self.temporal_cols:
            known_embedding_dict[col] = self.temporal_embedding_dict[col](known_input[col])
        ## 6.1 Positional Encoding, Modality Embedding
        for modality_idx, col in enumerate(["global_known"] + self.temporal_cols):
            embedding = torch.cat([cls_token, known_embedding_dict[col]], dim=1)
            pos_enc = self.pos_enc(embedding)
            mod_emb = self.mod_emb(torch.full(embedding.shape[:-1], modality_idx, device=embedding.device))
            known_embedding_dict[col] = embedding + pos_enc + mod_emb

        # 7. Temporal Block Decoding
        temporal_block_decoding = self.temporal_block_decoder(known_embedding_dict,
                                                              observed_encoding,
                                                              metadata_encoding,
                                                              text_encoding,
                                                              img_encoding,
                                                              known_mask,
                                                              observed_mask,
                                                              metadata_input["attention_mask"],
                                                              text_input["attention_mask"],
                                                              img_mask)

        # Loss
        output = self.output(temporal_block_decoding)[:, 1:, :] # Remove CLS token
        loss_raw = self.mse_loss(output, known_input[temporal_target_col]).squeeze()
        masked_loss = loss_raw * known_mask
        loss = masked_loss.sum() / known_mask.sum()

        return loss, output
