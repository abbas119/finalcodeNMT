# NMT_EEGPT_Project/models/nmt_eegpt_downstream_model.py
import torch
import torch.nn as nn
from .nmt_eegpt_pretrain_model import Pretrain_NMT_EEGPT # To load encoder part or full model

class NMT_EEGPT_Classifier(nn.Module):
    def __init__(self, 
                 # Params to reconstruct the pretrained encoder part of NMT_EEGPT_Pretrain
                 n_channels_model, segment_time_len_samples, patch_time_len_samples,
                 embed_dim, encoder_layers, num_heads, ff_dim, dropout_transformer, 
                 num_summary_tokens,
                 # Classifier head params
                 n_classes,
                 # Adaptive Spatial Filter params
                 use_adaptive_spatial_filter=True, 
                 n_input_channels_to_asf=21, # User's original number of channels
                 # Path to load pretrained weights
                 pretrained_encoder_path=None,
                 freeze_encoder=True 
                ):
        super().__init__()

        # Instantiate the NMT_EEGPT_Pretrain model to easily access its components
        # Or, you could have a separate EEGPT_Style_Encoder class that you load.
        # For simplicity, let's assume we load the whole Pretrain_NMT_EEGPT and use its feature extractor.
        self.feature_extractor_base = Pretrain_NMT_EEGPT( # Only to define structure
            n_channels_model=n_channels_model, # Channels model was PRETRAINED on (e.g. 19)
            segment_time_len_samples=segment_time_len_samples,
            patch_time_len_samples=patch_time_len_samples,
            embed_dim=embed_dim,
            encoder_layers=encoder_layers, 
            # These are for the online_encoder part:
            predictor_layers=1, reconstructor_layers=1, # Dummy for structure, not used in feature extraction
            num_heads=num_heads, ff_dim=ff_dim, dropout_transformer=dropout_transformer,
            num_summary_tokens=num_summary_tokens
        )

        if pretrained_encoder_path:
            try:
                checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
                # If checkpoint saves entire Pretrain_NMT_EEGPT:
                self.feature_extractor_base.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Loaded pretrained NMT-EEGPT weights from {pretrained_encoder_path}")
            except Exception as e:
                logging.error(f"Error loading pretrained NMT-EEGPT weights: {e}. Model will be randomly initialized.")
        else:
            logging.warning("No pretrained_encoder_path provided for NMT_EEGPT_Classifier. Encoder will be randomly initialized (not recommended for SSL approach).")

        if freeze_encoder:
            for param in self.feature_extractor_base.parameters(): # Freeze all initially
                param.requires_grad = False
            # Unfreeze only the online_encoder's summary_tokens and embedding_layer if fine-tuning those minimally
            # For strict linear probing, even these might be frozen.
            # For EEGPT's linear probing, encoder is frozen[cite: 464, 468].
            if hasattr(self.feature_extractor_base, 'online_encoder') and \
               hasattr(self.feature_extractor_base.online_encoder, 'summary_tokens'):
                self.feature_extractor_base.online_encoder.summary_tokens.requires_grad = True # Make summary tokens tunable

        self.use_asf = use_adaptive_spatial_filter
        self.n_input_channels_to_asf = n_input_channels_to_asf
        self.n_model_channels = n_channels_model # Channels the pretrained feature_extractor expects (e.g. 19)

        if self.use_asf:
            # ASF maps from original NMT channels (e.g. 21) to model's expected channels (e.g. 19)
            self.adaptive_spatial_filter = nn.Conv1d(
                self.n_input_channels_to_asf, 
                self.n_model_channels, 
                kernel_size=1, 
                bias=False # EEGPT ASF is 1x1 conv
            )
            if freeze_encoder == False : # If fine-tuning encoder, ASF should also be tunable
                 for param in self.adaptive_spatial_filter.parameters(): param.requires_grad = True
            else: # Linear probing, ASF is part of the new head.
                 for param in self.adaptive_spatial_filter.parameters(): param.requires_grad = True


        # Classifier head (operates on features from summary tokens)
        # Feature dim from encoder's summary tokens is (num_summary_tokens * embed_dim) if concatenated,
        # or just embed_dim if averaged or first one taken.
        # Assuming the .extract_features method returns (B, embed_dim)
        self.classifier_head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # Input x: (B, n_input_channels_to_asf, segment_time_len_samples)
        
        if self.use_asf:
            x = self.adaptive_spatial_filter(x) # Output: (B, n_model_channels, segment_time_len_samples)
        
        # x must now be converted to patches: (B, C_model, N_t, P_t)
        # This patching logic should be part of feature_extractor_base or handled here.
        # Let's assume feature_extractor_base.extract_features takes (B, C_model, T_segment)
        # and handles its own patching.
        
        # Patches for the feature extractor
        # (B, C_model, T_segment) -> (B, C_model, N_time_patches, patch_time_len_samples)
        B, C, T = x.shape
        P_t = self.feature_extractor_base.patch_time_len_samples
        N_t = T // P_t
        
        if T % P_t != 0:
            # This should not happen if input segments are correctly sized
            raise ValueError(f"Segment length {T} not divisible by patch time length {P_t}")
            
        x_segment_patches_raw = x.reshape(B, C, N_t, P_t)
        
        # Extract features using the (potentially frozen) pretrained encoder
        # The .extract_features method was defined in Pretrain_NMT_EEGPT
        features = self.feature_extractor_base.extract_features(x_segment_patches_raw) # (B, embed_dim)
        
        logits = self.classifier_head(features) # (B, n_classes)
        return logits

if __name__ == '__main__':
    # Example (requires config_ssl_pretrain and config_ssl_finetune)
    _N_CHANNELS_MODEL = 19 
    _SEGMENT_TIME_LEN_SAMPLES = 1024 # 4s * 256Hz
    _PATCH_TIME_LEN_SAMPLES = 64    # 250ms * 256Hz
    _EMBED_DIM = 256
    _ENCODER_LAYERS = 4
    _NUM_HEADS = 4
    _FF_DIM = _EMBED_DIM * 2
    _DROPOUT_TRANSFORMER = 0.1
    _NUM_SUMMARY_TOKENS = 1
    _N_CLASSES_DOWNSTREAM = 2
    _N_INPUT_CHANNELS_TO_ASF = 21 # User's original data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First, create a dummy pretrained model and save its state_dict (simulating pretraining)
    # This is just to test loading, actual pretraining is separate
    dummy_pretrain_model = Pretrain_NMT_EEGPT(
        n_channels_model=_N_CHANNELS_MODEL, segment_time_len_samples=_SEGMENT_TIME_LEN_SAMPLES,
        patch_time_len_samples=_PATCH_TIME_LEN_SAMPLES, embed_dim=_EMBED_DIM,
        encoder_layers=_ENCODER_LAYERS, predictor_layers=1, reconstructor_layers=1, # dummy
        num_heads=_NUM_HEADS, ff_dim=_FF_DIM, dropout_transformer=_DROPOUT_TRANSFORMER,
        num_summary_tokens=_NUM_SUMMARY_TOKENS
    )
    # Ensure the directory exists
    os.makedirs('models/saved_ssl_pretrain/', exist_ok=True)
    dummy_checkpoint_path = 'models/saved_ssl_pretrain/dummy_nmt_eegpt_encoder.pt'
    torch.save({'model_state_dict': dummy_pretrain_model.state_dict()}, dummy_checkpoint_path)
    print(f"Saved dummy pretrained model to {dummy_checkpoint_path}")

    # Now, instantiate the downstream classifier
    downstream_model = NMT_EEGPT_Classifier(
        n_channels_model=_N_CHANNELS_MODEL, # Channels the encoder core expects
        segment_time_len_samples=_SEGMENT_TIME_LEN_SAMPLES,
        patch_time_len_samples=_PATCH_TIME_LEN_SAMPLES,
        embed_dim=_EMBED_DIM,
        encoder_layers=_ENCODER_LAYERS,
        num_heads=_NUM_HEADS,
        ff_dim=_FF_DIM,
        dropout_transformer=_DROPOUT_TRANSFORMER,
        num_summary_tokens=_NUM_SUMMARY_TOKENS,
        n_classes=_N_CLASSES_DOWNSTREAM,
        use_adaptive_spatial_filter=True,
        n_input_channels_to_asf=_N_INPUT_CHANNELS_TO_ASF, # User's raw NMT data channel count
        pretrained_encoder_path=dummy_checkpoint_path, # Path to saved Pretrain_NMT_EEGPT model
        freeze_encoder=True # For linear probing
    ).to(device)

    param_count_total = sum(p.numel() for p in downstream_model.parameters())
    param_count_trainable = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
    print(f"NMT_EEGPT_Classifier initialized. Total Params: {param_count_total/1e6:.2f}M, Trainable Params: {param_count_trainable/1e3:.2f}K")

    # Test with dummy input (Batch, UserChannels=21, TimeSamples)
    dummy_input_downstream = torch.randn(4, _N_INPUT_CHANNELS_TO_ASF, _SEGMENT_TIME_LEN_SAMPLES).to(device)
    output = downstream_model(dummy_input_downstream)
    print(f"Downstream model output shape: {output.shape}") # Expected: (4, 2)