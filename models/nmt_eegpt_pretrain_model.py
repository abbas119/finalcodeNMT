# NMT_EEGPT_Project/models/nmt_eegpt_pretrain_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nmt_eegpt_blocks import LocalSpatioTemporalEmbedding, \
                              EEGPT_Style_Encoder, EEGPT_Style_Predictor, EEGPT_Style_Reconstructor
import random

class NMT_EEGPT_Pretrain(nn.Module):
    def __init__(self, 
                 # Data/Patching params
                 n_channels_model, # e.g., 19
                 segment_time_len_samples, # e.g., 4s * 256Hz = 1024
                 patch_time_len_samples,   # e.g., 250ms * 256Hz = 64
                 # Model architecture params (should match config_ssl_pretrain)
                 embed_dim, 
                 encoder_layers, predictor_layers, reconstructor_layers,
                 num_heads, ff_dim, dropout_transformer, 
                 num_summary_tokens=1, # S in EEGPT
                 # Masking params
                 time_patch_mask_percentage=0.5,  # EEGPT: 50% time patch masking [cite: 481]
                 channel_mask_percentage=0.8,     # EEGPT: 80% channel masking [cite: 481]
                 # Momentum encoder
                 momentum_tau=0.01 # EEGPT accumulation factor
                ):
        super().__init__()
        
        self.n_channels_model = n_channels_model
        self.segment_time_len_samples = segment_time_len_samples
        self.patch_time_len_samples = patch_time_len_samples
        self.num_time_patches_per_channel = segment_time_len_samples // patch_time_len_samples
        self.total_patches_per_segment = self.n_channels_model * self.num_time_patches_per_channel
        self.embed_dim = embed_dim
        self.num_summary_tokens = num_summary_tokens

        self.time_patch_mask_percentage = time_patch_mask_percentage
        self.channel_mask_percentage = channel_mask_percentage

        # 1. Local Spatio-Temporal Embedding Layer
        self.embedding_layer = LocalSpatioTemporalEmbedding(
            n_channels_model=self.n_channels_model,
            num_time_patches_per_channel=self.num_time_patches_per_channel, # Not directly used by LSTE if it takes flat patches
            patch_time_len_samples=self.patch_time_len_samples,
            embed_dim=self.embed_dim,
            pos_embed_dropout=dropout_transformer # Use same dropout
        )

        # 2. Online Network Components
        # Encoder: Processes unmasked patches (content from visible parts) + summary tokens
        self.online_encoder = EEGPT_Style_Encoder(
            embed_dim, encoder_layers, num_heads, ff_dim, dropout_transformer, num_summary_tokens
        )
        
        # Predictor: Uses encoder output + query tokens to predict features for MASKED patches
        # Num query tokens should correspond to the max number of *truly masked* patches
        # This is complex. EEGPT: "learnable vector query is used as the query token" for M_bar [cite: 442]
        # For MAE style, predictor reconstructs masked tokens.
        # For EEGPT alignment, predictor predicts *features* of masked tokens.
        # Let's assume predictor aims to output features for *all* patch positions.
        self.online_predictor = EEGPT_Style_Predictor(
            embed_dim, predictor_layers, num_heads, ff_dim, dropout_transformer,
            num_query_tokens_for_masked=0 # EEGPT Fig 1 suggests predictor takes encoder output + pos and outputs for all time
        )

        # Reconstructor: Uses encoder output (for unmasked) + predictor output (for masked) to reconstruct raw MASKED patches
        self.online_reconstructor = EEGPT_Style_Reconstructor(
            embed_dim, reconstructor_layers, num_heads, ff_dim, dropout_transformer,
            original_patch_dim=self.patch_time_len_samples
        )

        # 3. Momentum Network Components (only Encoder)
        self.momentum_tau = momentum_tau
        self.momentum_encoder = EEGPT_Style_Encoder( # Same architecture as online_encoder
            embed_dim, encoder_layers, num_heads, ff_dim, dropout_transformer, num_summary_tokens
        )
        self._init_momentum_encoder()

        # Learnable MASK token embedding (for replacing content of masked patches before encoder)
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Learnable query tokens for predictor (if needed, for predicting M_bar features)
        # This is part of EEGPT's predictor mechanism for "unseen" parts by the encoder.
        # Let's assume for now predictor takes full sequence of (potentially placeholder) tokens.
        self.predictor_query_tokens = nn.Parameter(torch.randn(1, self.total_patches_per_segment, embed_dim))


    @torch.no_grad()
    def _init_momentum_encoder(self):
        for param_online, param_momentum in zip(self.online_encoder.parameters(), self.momentum_encoder.parameters()):
            param_momentum.data.copy_(param_online.data)
            param_momentum.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_online, param_momentum in zip(self.online_encoder.parameters(), self.momentum_encoder.parameters()):
            param_momentum.data = param_momentum.data * (1.0 - self.momentum_tau) + param_online.data * self.momentum_tau

    def _generate_masks(self, batch_size, device):
        # EEGPT: "masking (50% time and 80% channel patches)" [cite: 435]
        # "The encoder processes the masked part, extracting features (enc_j)" - this implies encoder sees UNMASKED patches.
        # "The predictor predicts features (pred_j) for all time segments, aligning with Momentum Encoder output (menc_j)"
        # This means predictor must predict features for the MASKED locations.

        # Masking for spatio-temporal representation alignment (L_A) and mask-based reconstruction (L_R)
        # We need a boolean mask (B, C, N_t) where True means the patch p_c,t is masked (its content hidden).
        
        # 1. Time masking: mask 50% of time patch indices (columns in C x N_t grid)
        num_time_patches_to_mask = int(self.num_time_patches_per_channel * self.time_patch_mask_percentage)
        masked_time_indices = torch.zeros(batch_size, self.num_time_patches_per_channel, dtype=torch.bool, device=device)
        for i in range(batch_size):
            indices = torch.randperm(self.num_time_patches_per_channel, device=device)[:num_time_patches_to_mask]
            masked_time_indices[i, indices] = True # True if time index is masked

        # 2. Channel masking: mask 80% of channel indices (rows in C x N_t grid)
        num_channels_to_mask = int(self.n_channels_model * self.channel_mask_percentage)
        masked_channel_indices = torch.zeros(batch_size, self.n_channels_model, dtype=torch.bool, device=device)
        for i in range(batch_size):
            indices = torch.randperm(self.n_channels_model, device=device)[:num_channels_to_mask]
            masked_channel_indices[i, indices] = True # True if channel index is masked

        # Combine: A patch (c,t) is "conceptually masked" if its channel OR its time is selected by these masks.
        # The final mask for MAE-style input to encoder would be where actual patch content is replaced.
        # For EEGPT's dual task, the interpretation of "masked part M" and "unmasked part M_bar" is key.
        # M_bar is the part whose features are predicted by Predictor and raw signal by Reconstructor.
        # M is the part visible to the Encoder.
        
        # Create a flat binary mask (B, TotalPatches) - True if patch is to be hidden from encoder / reconstructed
        # This is the "masked part M_bar" in EEGPT terminology.
        final_patch_mask_2d = masked_channel_indices.unsqueeze(2) | masked_time_indices.unsqueeze(1) # (B, C, N_t)
        final_patch_mask_flat = rearrange(final_patch_mask_2d, 'b c nt -> b (c nt)') # (B, TotalPatches)
        
        # Indices of patches in M_bar (truly masked, for loss calculation)
        # Indices of patches in M (visible to encoder)
        return final_patch_mask_flat # True means masked (target for prediction/reconstruction)

    def forward_pretrain(self, x_segment_patches_raw):
        # x_segment_patches_raw: (B, C, N_time_patches, patch_time_len_samples) - from dataset_ssl
        B, C, N_t, P_t = x_segment_patches_raw.shape
        device = x_segment_patches_raw.device

        # Flatten patches for embedding and sequence processing
        # (B, C, N_t, P_t) -> (B, C*N_t, P_t)
        x_patches_flat_for_embed = rearrange(x_segment_patches_raw, 'b c nt pt -> b (c nt) pt')
        
        # Channel IDs for embedding: (B, C*N_t)
        channel_ids_per_patch = torch.arange(C, device=device).repeat_interleave(N_t)
        channel_ids_flat = channel_ids_per_patch.unsqueeze(0).expand(B, -1)
        
        # 1. Embed all original patches
        all_embedded_tokens = self.embedding_layer(x_patches_flat_for_embed, channel_ids_flat) # (B, TotalPatches, EmbedDim)
                                                                                                # This includes content+channel+pos

        # 2. Generate mask: `patch_mask_flat` is True for patches in M_bar (masked for encoder input)
        patch_mask_flat = self._generate_masks(B, device) # (B, TotalPatches)

        # 3. Prepare input for Online Encoder
        # Replace content of M_bar patches with a learnable MASK token embedding
        # Unmasked patches (M) keep their original embedded content.
        input_to_online_encoder = torch.where(
            patch_mask_flat.unsqueeze(-1), # (B, TotalPatches, 1) -> expand to EmbedDim
            self.mask_token_embed.expand(B, self.total_patches_per_segment, -1), # Broadcast MASK token
            all_embedded_tokens # Original embedded tokens
        )
        
        # 4. Online Encoder processing (gets M (unmasked content) + MASK tokens for M_bar) [cite: 434, 436]
        # Note: EEGPT states Encoder gets "masked patches", processes "masked part M".
        # This usually means it sees the *unmasked* portions. Fig 1 description: "encoder processes the masked part,
        # extracting features (enc_j) ... for each time segment in the M part". M is unmasked visible part.
        # So, input_to_online_encoder should represent the *visible* data.
        # Let's refine: encoder gets only *unmasked* tokens + summary tokens.
        # This is a common MAE setup. EEGPT's Fig 1 is a bit abstract here.
        # Let's follow a simpler MAE-style for encoder input: non-masked tokens + summary.
        # The "dual SSL" happens at the loss stage.

        # Alternative input to encoder (MAE style - only unmasked tokens):
        # This requires gathering, then passing to encoder, then scattering for predictor/reconstructor.
        # For EEGPT's hierarchical structure (spatial ENC then temporal PRED/REC), it's more complex.
        # The current `EEGPT_Style_Encoder` takes a sequence.
        # Let's assume `input_to_online_encoder` is all_embedded_tokens but with MASK token for masked patches.
        
        # Add summary tokens for the online encoder
        summary_tokens_batch = self.online_encoder.summary_tokens.expand(B, -1, -1)
        encoder_input_seq = torch.cat((summary_tokens_batch, input_to_online_encoder), dim=1)
        
        # enc_j_plus_summary: Output of online encoder, features for summary + all patch positions (some were MASK tokens)
        enc_j_plus_summary = self.online_encoder(encoder_input_seq) # (B, S + TotalPatches, EmbedDim)
        enc_j_patches = enc_j_plus_summary[:, self.num_summary_tokens:, :] # (B, TotalPatches, EmbedDim)
                                                                    # These are features after encoder saw input with MASK tokens

        # 5. Online Predictor
        # Predictor uses `enc_j_patches` (which contains info from unmasked parts and processed MASK tokens)
        # It should predict features for ALL patches, especially improving representation at MASKED locations.
        # EEGPT: PRED uses enc_j + pos_j from M part, and query for M_bar.
        # Simpler: PRED refines all enc_j_patches.
        # For EEGPT, the predictor input could be `enc_j_patches` where masked positions get `predictor_query_tokens`.
        # predictor_input_tokens = torch.where(
        #    patch_mask_flat.unsqueeze(-1),
        #    self.predictor_query_tokens.expand(B, -1, -1), # Query for masked
        #    enc_j_patches # Features from encoder for unmasked
        # )
        # Or simply pass all `enc_j_patches` to the predictor to refine them
        predicted_features_all_patches = self.online_predictor(enc_j_patches) # (B, TotalPatches, EmbedDim)
        
        # Select predicted features corresponding to the MASKED patches for Alignment Loss (L_A)
        # (B, TotalPatches, EmbedDim) -> select based on patch_mask_flat
        # Need to reshape patch_mask_flat to gather properly or use boolean indexing carefully
        predicted_features_for_LA = predicted_features_all_patches[patch_mask_flat] # (Num_Truly_Masked_Patches_In_Batch, EmbedDim)


        # 6. Momentum Encoder Branch (for L_A targets)
        with torch.no_grad():
            self._update_momentum_encoder() # Update momentum encoder
            # Momentum encoder sees all ORIGINAL UNMASKED embedded tokens + summary tokens
            momentum_encoder_input_seq = torch.cat((summary_tokens_batch, all_embedded_tokens), dim=1) # Original, no MASK tokens
            menc_j_plus_summary = self.momentum_encoder(momentum_encoder_input_seq)
            menc_j_patches = menc_j_plus_summary[:, self.num_summary_tokens:, :] # (B, TotalPatches, EmbedDim)
            
            # Target for L_A: features from momentum encoder for the MASKED patches
            target_features_for_LA = menc_j_patches[patch_mask_flat].detach() # (Num_Truly_Masked_Patches_In_Batch, EmbedDim)


        # 7. Online Reconstructor (for L_R targets)
        # Input: Features from online_encoder for UNMASKED parts (M),
        #        Features from online_predictor for MASKED parts (M_bar).
        #        And positional information.
        # This means we need to combine `enc_j_patches` (for unmasked regions)
        # and `predicted_features_all_patches` (for masked regions).
        reconstructor_input_features = torch.where(
            patch_mask_flat.unsqueeze(-1),
            predicted_features_all_patches, # Use predictor's output for masked regions
            enc_j_patches                   # Use encoder's direct output for unmasked regions (skip connection)
        )
        
        reconstructed_token_embeddings = self.online_reconstructor(reconstructor_input_features) # (B, TotalPatches, EmbedDim)
        # Reconstruct raw patch values ONLY for MASKED patches
        reconstructed_raw_patches_for_LR = self.online_reconstructor.to_raw_patch_projection(
            reconstructed_token_embeddings[patch_mask_flat] # (Num_Truly_Masked_Patches_In_Batch, EmbedDim)
        ) # (Num_Truly_Masked_Patches_In_Batch, PatchTimeLen)

        # Target for L_R: original raw values of MASKED patches
        target_raw_patches_for_LR = x_patches_flat_for_embed[patch_mask_flat] # (Num_Truly_Masked_Patches_In_Batch, PatchTimeLen)

        return predicted_features_for_LA, reconstructed_raw_patches_for_LR, \
               target_features_for_LA, target_raw_patches_for_LR

    def extract_features(self, x_segment_patches_raw):
        # For downstream tasks: use the online_encoder (after pretraining)
        # x_segment_patches_raw: (B, C, N_t, P_t)
        B, C, N_t, P_t = x_segment_patches_raw.shape
        device = x_segment_patches_raw.device

        x_patches_flat_for_embed = rearrange(x_segment_patches_raw, 'b c nt pt -> b (c nt) pt')
        channel_ids_per_patch = torch.arange(C, device=device).repeat_interleave(N_t)
        channel_ids_flat = channel_ids_per_patch.unsqueeze(0).expand(B, -1)
        
        all_embedded_tokens = self.embedding_layer(x_patches_flat_for_embed, channel_ids_flat)
        
        summary_tokens_batch = self.online_encoder.summary_tokens.expand(B, -1, -1)
        encoder_input_seq = torch.cat((summary_tokens_batch, all_embedded_tokens), dim=1)
        
        encoded_output = self.online_encoder(encoder_input_seq) # (B, S + TotalPatches, EmbedDim)
        
        # Use summary tokens as the representation [cite: 466]
        features = encoded_output[:, :self.num_summary_tokens, :] # (B, S, EmbedDim)
        # Example: average summary tokens if S > 1, or take first one if S=1
        return features.mean(dim=1) # (B, EmbedDim)