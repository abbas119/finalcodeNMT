# NMT_EEGPT_Project/models/nmt_eegpt_blocks.py
import torch
import torch.nn as nn
from einops import rearrange

# Rotary Positional Embedding (RoPE)
# You can find several implementations online, e.g., from lucidrains/rotary-embedding-torch
# Or a simplified version if full RoPE is too complex initially.
# For this example, I'll use basic learnable embeddings, but RoPE is recommended for better performance.
# If using RoPE, it's typically applied within the attention mechanism of the Transformer block.

class LocalSpatioTemporalEmbedding(nn.Module):
    """
    Implements EEGPT's local spatio-temporal embedding (Section 2.3).
    Input: Raw patches (B, NumTotalPatches, PatchTimeLen)
           Channel IDs (B, NumTotalPatches) - indicating original channel of each patch
           Time Patch IDs (B, NumTotalPatches) - indicating original time patch index
    Output: Embedded tokens (B, NumTotalPatches, EmbedDim)
    """
    def __init__(self, n_channels_model, num_time_patches_per_channel, 
                 patch_time_len_samples, embed_dim, pos_embed_dropout=0.1):
        super().__init__()
        self.patch_linear_projector = nn.Linear(patch_time_len_samples, embed_dim)
        self.channel_embed = nn.Embedding(n_channels_model, embed_dim)
        
        # For simplicity using separate learnable positional embeddings for time aspect of patches
        # EEGPT mentions RoPE[cite: 441], which is more complex and applied differently.
        # A simpler learnable pos embedding for the sequence of N_total_patches
        self.total_patches_per_segment = n_channels_model * num_time_patches_per_channel
        self.temporal_patch_sequence_pos_embed = nn.Parameter(
            torch.randn(1, self.total_patches_per_segment, embed_dim)
        )
        # A more EEGPT-like approach combines patch content + channel embedding.
        # Temporal position (pos_j) is added to enc_j and pred_j later.
        # Rotary pos embedding (RoPE) is applied within attention layers usually.

        self.dropout = nn.Dropout(pos_embed_dropout)

    def forward(self, x_patches_flat, channel_ids_flat):
        # x_patches_flat: (B, TotalPatches, PatchTimeLen)
        # channel_ids_flat: (B, TotalPatches) - integer id for channel of each patch
        
        # 1. Linear embed patch content [cite: 458]
        patch_content_embed = self.patch_linear_projector(x_patches_flat) # (B, TotalPatches, EmbedDim)
        
        # 2. Add channel embedding (zeta_i in EEGPT Eq. 11 [cite: 459])
        ch_embeds = self.channel_embed(channel_ids_flat) # (B, TotalPatches, EmbedDim)
        
        tokens = patch_content_embed + ch_embeds
        
        # Add learnable positional embedding for the whole sequence of patches
        # This is a simplification of EEGPT's pos_j and RoPE.
        tokens = tokens + self.temporal_patch_sequence_pos_embed[:, :tokens.size(1), :]
        
        return self.dropout(tokens)


class NMT_EEGPT_TransformerModule(nn.Module):
    """A standard Transformer Encoder module."""
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_transformer,
            batch_first=True,
            norm_first=True # Generally recommended for Transformer stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_tokens_with_pos): # Assumes positional encoding is already part of x_tokens
        return self.transformer_encoder(x_tokens_with_pos)


# --- EEGPT Specific Modules: Encoder, Predictor, Reconstructor ---
# These are all Transformer-based according to EEGPT paper (Section 2.1, 2.2, Fig 1)
# They operate on sequences of embedded tokens.

class EEGPT_Style_Encoder(NMT_EEGPT_TransformerModule): # Inherits from the generic one
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, num_summary_tokens=1):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        self.num_summary_tokens = num_summary_tokens
        if num_summary_tokens > 0:
            self.summary_tokens = nn.Parameter(torch.randn(1, num_summary_tokens, embed_dim))

    def forward(self, x_tokens): # x_tokens: (B, SeqLen_of_unmasked_patches, EmbedDim)
        B = x_tokens.shape[0]
        if self.num_summary_tokens > 0:
            summary_tokens_batch = self.summary_tokens.expand(B, -1, -1)
            x_input = torch.cat((summary_tokens_batch, x_tokens), dim=1)
        else:
            x_input = x_tokens
        
        encoded_output = super().forward(x_input) # (B, S+SeqLen, EmbedDim)
        
        # Separate summary token features and patch features if needed for specific logic
        # For EEGPT: encoder processes masked patches (actually unmasked patch content + learnable MASK tokens)
        # And produces enc_j for those unmasked parts, and summary token outputs.
        # This forward needs to be aligned with how EEGPT structures its inputs/outputs for "enc_j"
        return encoded_output # This now contains features for summary tokens + input patches

class EEGPT_Style_Predictor(NMT_EEGPT_TransformerModule):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, num_query_tokens_for_masked):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        # Query tokens are learnable vectors used to prompt prediction for masked parts [cite: 442]
        self.num_query_tokens = num_query_tokens_for_masked
        if self.num_query_tokens > 0:
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens_for_masked, embed_dim))

    def forward(self, x_encoded_unmasked_tokens_with_pos):
        # x_encoded_unmasked_tokens_with_pos: (B, SeqLen_unmasked_plus_S, EmbedDim) (output from Encoder)
        # EEGPT: Predictor uses features enc_j from Encoder (for M part) + pos_j
        #        and query tokens for predicting features belonging to M_bar (unmasked part)
        # This requires careful concatenation of encoder's output (for unmasked) and query tokens (for masked).
        # For simplicity here, if just passing encoder output:
        # predicted_features = super().forward(x_encoded_unmasked_tokens_with_pos)
        # If adding query tokens:
        B = x_encoded_unmasked_tokens_with_pos.shape[0]
        if self.num_query_tokens > 0:
            query_tokens_batch = self.query_tokens.expand(B, -1, -1)
            # The input to predictor should be features of UNMASKED parts + query tokens for MASKED parts
            # This structure depends on how masking is handled and inputs are prepared.
            # A common way: concatenate features of unmasked known context + query tokens
            # predictor_input = torch.cat((x_encoded_unmasked_tokens_with_pos, query_tokens_batch), dim=1)
            # Then PRED processes this to give {pred_t} for all positions.
            # For now, this forward is conceptual based on general use of query tokens.
            # The actual input would be structured based on EEGPT's Fig 1 and equations.
            # Example: just processes what's given
            return super().forward(x_encoded_unmasked_tokens_with_pos) # Placeholder
        else:
            return super().forward(x_encoded_unmasked_tokens_with_pos)


class EEGPT_Style_Reconstructor(NMT_EEGPT_TransformerModule):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout_transformer, original_patch_dim):
        super().__init__(embed_dim, num_layers, num_heads, ff_dim, dropout_transformer)
        self.to_raw_patch_projection = nn.Linear(embed_dim, original_patch_dim) # To reconstruct patch values

    def forward(self, x_combined_features_for_reconstruction_with_pos):
        # x_combined: Features from Encoder (unmasked part) AND Predictor (masked part) + pos_j
        # This implies a sequence of all tokens (some original, some predicted) is fed here.
        reconstructed_token_embeddings = super().forward(x_combined_features_for_reconstruction_with_pos)
        reconstructed_raw_patches = self.to_raw_patch_projection(reconstructed_token_embeddings)
        return reconstructed_raw_patches # (B, TotalPatches_in_sequence, original_patch_dim)