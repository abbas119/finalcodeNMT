# NMT_EEGPT_Project/config_ssl_pretrain.py
import torch
import os

# --- Dataset Paths ---
# Uses ALL NMT data (normal & abnormal) for pretraining.
# Assumes dataset_utils.py has processed EDFs into 4s segments and saved them here.
PREPROCESSED_SSL_DATA_DIR = 'data/processed_nmt_ssl/' # All 4s segments for SSL
os.makedirs(PREPROCESSED_SSL_DATA_DIR, exist_ok=True)
# Subfolders for normal/abnormal might not be needed if dataset_ssl.py just lists all segment files.

# --- Data Parameters (used by dataset_ssl.py and model) ---
# These should largely match config_supervised for consistency in signal properties
N_CHANNELS_MODEL = 19         # Number of channels the NMT-EEGPT model will internally process
TARGET_SFREQ = 256            # Hz, align with EEGPT [cite: 476]
SEGMENT_DURATION_SEC = 4.0    # s, EEGPT uses 4s crops [cite: 476]
INPUT_TIME_LENGTH_MODEL = int(TARGET_SFREQ * SEGMENT_DURATION_SEC) # Samples per 4s segment

# Patching (Inspired by EEGPT Section 2.3, Table 12, 13 [cite: 747, 749])
# EEGPT uses 250ms patches (d=64 at 256Hz) for 4s input (T=1024)
PATCH_DURATION_MS = 250
PATCH_TIME_LENGTH_SAMPLES = int(PATCH_DURATION_MS / 1000 * TARGET_SFREQ) # e.g., 64 samples
N_TIME_PATCHES = INPUT_TIME_LENGTH_MODEL // PATCH_TIME_LENGTH_SAMPLES # e.g., 1024 // 64 = 16

# Masking Strategy (EEGPT: 50% time, 80% channel patches [cite: 435, 481])
# This usually means masking 50% of the *temporal patch indices*
# And, for the *channel dimension*, masking 80% of the *channel indices*.
# A patch p_c,t is considered masked if EITHER its channel 'c' OR its time index 't' is masked.
# Or it can mean a certain fraction of (channel, time_patch_index) pairs are masked.
# Let's define fraction of total patches to mask for reconstruction.
# The dual SSL task in EEGPT has specific roles for masked/unmasked parts for Encoder vs Predictor.
MASK_PATCH_PERCENTAGE_FOR_RECON = 0.75 # BERT default, MAE default. EEGPT is more complex.
                                      # EEGPT: "50% time and 80% channel patches" - this is high.
                                      # Let's aim for a high overall masking ratio if following MAE.
                                      # If following EEGPT's dual task, this needs specific setup for L_A and L_R.
                                      # For now, this is a general masking ratio for MAE-like reconstruction.

# --- NMT-EEGPT Model Parameters (Scaled for RTX 3060, e.g., EEGPT-Tiny/Base like) ---
# EEGPT Tiny variants (0.4M-1.6M), Little (6.4M), Base1/2 (19M/25M), Large (101M) [cite: 519]
# Let's aim for a model around 5-10M parameters.
# Example: Based on EEGPT Base1 (19M, de=256, L=6/6/6, S=1) or Base2 (25M, de=256, L=8/8/8, S=4)
# We might need to scale down embed_dim or layers.
EMBED_DIM = 256               # de in EEGPT [cite: 519] (e.g., 64 for tiny, 256 for base)
ENCODER_LAYERS = 4            # Number of Transformer blocks in Encoder [cite: 519] (e.g., 2, 6, or 8)
PREDICTOR_LAYERS = 2          # Number of Transformer blocks in Predictor [cite: 519]
RECONSTRUCTOR_LAYERS = 2      # Number of Transformer blocks in Reconstructor [cite: 519]
NUM_HEADS = 4                 # Attention heads (e.g., 4 or 8)
FEEDFORWARD_DIM = EMBED_DIM * 2 # Or *4. Hidden dim of FFN in Transformer blocks
DROPOUT_PRETRAIN = 0.1        # Dropout rate during pretraining
NUM_SUMMARY_TOKENS = 1        # S in EEGPT (e.g., 1 or 4) [cite: 519]
MOMENTUM_TAU = 0.01           # Accumulation factor for momentum encoder (EEGPT value)

# --- Pretraining Parameters ---
INIT_LR_PRETRAIN = 5e-4       # EEGPT uses 5e-4 max with OneCycle [cite: 483]
BATCH_SIZE_PRETRAIN = 16      # Adjust based on GPU memory (RTX 3060 12GB)
GRAD_ACCUMULATION_STEPS = 4   # Effective batch size = BATCH_SIZE_PRETRAIN * GRAD_ACCUMULATION_STEPS (e.g., 16*4=64)
MAX_EPOCHS_PRETRAIN = 100     # EEGPT: 200 epochs. Start with fewer.
OPTIMIZER_NAME = 'AdamW'
WEIGHT_DECAY_PRETRAIN = 0.05
LR_SCHEDULER_PRETRAIN = 'OneCycleLR' # EEGPT uses OneCycle [cite: 483]
MAX_LR_ONE CYCLE = INIT_LR_PRETRAIN
PCT_START_ONE_CYCLE = 0.3

CUDA = torch.cuda.is_available()
USE_AMP = True                # Use Automatic Mixed Precision if CUDA available

# Logging and Saving
LOG_DIR_SSL_PRETRAIN = 'logs/ssl_pretrain/'
MODEL_SAVE_DIR_SSL_PRETRAIN = 'models/saved_ssl_pretrain/'
PRETRAIN_SAVE_EVERY_EPOCHS = 5
os.makedirs(LOG_DIR_SSL_PRETRAIN, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR_SSL_PRETRAIN, exist_ok=True)