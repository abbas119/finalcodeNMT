# NMT_EEGPT_Project/pretrain_nmt_eegpt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import time
import random
from einops import rearrange
from tqdm import tqdm

import config_ssl_pretrain as cfg_ssl
from dataset_ssl import NMT_SSL_Patched_Dataset
from models.nmt_eegpt_pretrain_model import NMT_EEGPT_Pretrain
from monitors import TrainingMonitor # Using the enhanced monitor
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast

# Setup logger uniquely for this module
logger_ssl_pretrain = logging.getLogger('ssl_pretrain_logger')
# ... (similar logger setup as in train_supervised_baselines.py, using a unique name) ...
if not logger_ssl_pretrain.handlers:
    logger_ssl_pretrain.setLevel(logging.INFO)
    ch_ssl_pretrain = logging.StreamHandler()
    ch_ssl_pretrain.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s'))
    logger_ssl_pretrain.addHandler(ch_ssl_pretrain)


def setup_logging_ssl(log_dir, model_name_str="NMT_EEGPT_Pretrain"): # File logger
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'pretraining_log_{model_name_str}.txt')
    
    file_logger = logging.getLogger(f'{model_name_str}_file_logger')
    if file_logger.hasHandlers(): file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a') # Append if resuming
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    file_logger.addHandler(fh)
    return file_logger

# ... (generate_random_masks_for_batch function from previous pretrain_nmt_eegpt.py) ...
def generate_random_masks_for_batch(batch_size, n_channels, n_time_patches, 
                                   time_mask_percentage, channel_mask_percentage, device):
    num_time_patches_to_mask = int(n_time_patches * time_mask_percentage)
    masked_time_cols = torch.zeros(batch_size, n_time_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        indices = torch.randperm(n_time_patches, device=device)[:num_time_patches_to_mask]
        masked_time_cols[i, indices] = True 
    num_channels_to_mask = int(n_channels * channel_mask_percentage)
    masked_channel_rows = torch.zeros(batch_size, n_channels, dtype=torch.bool, device=device)
    for i in range(batch_size):
        indices = torch.randperm(n_channels, device=device)[:num_channels_to_mask]
        masked_channel_rows[i, indices] = True 
    m_bar_mask_2d = masked_channel_rows.unsqueeze(2) | masked_time_cols.unsqueeze(1)
    m_bar_mask_flat = rearrange(m_bar_mask_2d, 'b c nt -> b (c nt)')
    return m_bar_mask_flat


def pretrain_nmt_eegpt():
    current_run_logger = setup_logging_ssl(cfg_ssl.LOG_DIR_SSL_PRETRAIN)
    current_run_logger.info(f"--- Initializing NMT-EEGPT Self-Supervised Pretraining ---")
    current_run_logger.info(f"Using device: {'cuda' if cfg_ssl.CUDA else 'cpu'}")
    if cfg_ssl.USE_AMP and not cfg_ssl.CUDA:
        current_run_logger.warning("AMP can only be used with CUDA. Disabling AMP.")
        cfg_ssl.USE_AMP = False

    # --- Data Loading ---
    current_run_logger.info("Loading SSL dataset (all NMT segments)...")
    ssl_dataset = NMT_SSL_Patched_Dataset(
        data_dir_ssl=cfg_ssl.PREPROCESSED_SSL_DATA_DIR,
        segment_duration_sec=cfg_ssl.SEGMENT_DURATION_SEC,
        target_sfreq=cfg_ssl.TARGET_SFREQ,
        patch_duration_ms=cfg_ssl.PATCH_DURATION_MS,
        n_channels=cfg_ssl.N_CHANNELS_MODEL,
        test_mode_reduce_data=cfg_ssl.TEST_MODE_REDUCE_DATA if hasattr(cfg_ssl, 'TEST_MODE_REDUCE_DATA') else False,
        n_segments_test_mode=100 if hasattr(cfg_ssl, 'TEST_MODE_REDUCE_DATA') else 0)
    ssl_loader = DataLoader(ssl_dataset, batch_size=cfg_ssl.BATCH_SIZE_PRETRAIN, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    current_run_logger.info(f"SSL Dataset loaded. Total 4s segments for pretraining: {len(ssl_dataset)}")
    if len(ssl_dataset) == 0: current_run_logger.error("SSL dataset is empty. Exiting."); return

    # --- Model Initialization ---
    device = torch.device('cuda' if cfg_ssl.CUDA else 'cpu')
    model = NMT_EEGPT_Pretrain( # Changed from Pretrain_NMT_EEGPT to match class name
        n_channels_model=cfg_ssl.N_CHANNELS_MODEL,
        segment_time_len_samples=cfg_ssl.INPUT_TIME_LENGTH_MODEL,
        patch_time_len_samples=cfg_ssl.PATCH_TIME_LENGTH_SAMPLES,
        embed_dim=cfg_ssl.EMBED_DIM, encoder_layers=cfg_ssl.ENCODER_LAYERS,
        predictor_layers=cfg_ssl.PREDICTOR_LAYERS, reconstructor_layers=cfg_ssl.RECONSTRUCTOR_LAYERS,
        num_heads=cfg_ssl.NUM_HEADS, ff_dim=cfg_ssl.FEEDFORWARD_DIM,
        dropout_transformer=cfg_ssl.DROPOUT_PRETRAIN,
        num_summary_tokens=cfg_ssl.NUM_SUMMARY_TOKENS, momentum_tau=cfg_ssl.MOMENTUM_TAU
    ).to(device)
    current_run_logger.info(f"NMT-EEGPT Pretrain Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- Loss & Optimizer ---
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg_ssl.INIT_LR_PRETRAIN, weight_decay=cfg_ssl.WEIGHT_DECAY_PRETRAIN)
    scaler = GradScaler() if cfg_ssl.USE_AMP else None
    scheduler = None
    if cfg_ssl.LR_SCHEDULER_PRETRAIN == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg_ssl.MAX_LR_ONE_CYCLE, epochs=cfg_ssl.MAX_EPOCHS_PRETRAIN, 
            steps_per_epoch=len(ssl_loader), pct_start=cfg_ssl.PCT_START_ONE_CYCLE)
    # Add other schedulers if needed

    # --- Training Monitor & Checkpointing Setup ---
    model_name_str_pretrain = "NMT_EEGPT_Pretrain" # For monitor paths
    # Monitor total loss for SSL pretraining. Lower is better.
    monitor_metric_to_use_ssl = 'total_loss' # Custom for pretraining loop
    metric_mode_to_use_ssl = 'min'
    
    # Path for best pretraining model (based on lowest total loss, not typical for SSL but useful for tracking)
    # Usually, for SSL, we save periodic checkpoints and pick one based on downstream performance.
    # For simplicity, let's still define a "best" based on SSL loss.
    model_save_path_ssl = os.path.join(cfg_ssl.MODEL_SAVE_DIR_SSL_PRETRAIN, f'{model_name_str_pretrain}_best_ssl_loss.pt')
    checkpoint_dir_path_ssl = os.path.join(cfg_ssl.MODEL_SAVE_DIR_SSL_PRETRAIN, f'{model_name_str_pretrain}_checkpoints/')
    
    monitor_ssl = TrainingMonitor(
        model_path=model_save_path_ssl, 
        checkpoint_dir=checkpoint_dir_path_ssl, 
        patience=cfg_ssl.PATIENCE_PRETRAIN if hasattr(cfg_ssl, 'PATIENCE_PRETRAIN') else 50, # SSL might need more patience
        monitor_metric_name=monitor_metric_to_use_ssl, # Monitor 'total_loss'
        metric_mode=metric_mode_to_use_ssl  # 'min' for loss
    )
    
    start_epoch_ssl = 0

    # <<< --- ATTEMPT TO LOAD LATEST PRETRAIN CHECKPOINT --- >>>
    if os.path.isdir(checkpoint_dir_path_ssl) and any(f.startswith('checkpoint_epoch_') for f in os.listdir(checkpoint_dir_path_ssl)):
        current_run_logger.info(f"Attempting to load latest SSL pretrain checkpoint from {checkpoint_dir_path_ssl}...")
        resume_data_ssl = monitor_ssl.load_latest_checkpoint(model, optimizer)
        if resume_data_ssl:
            _model_loaded, _optimizer_loaded, loaded_epoch_num, best_metric_val_resumed, counter_resumed = resume_data_ssl
            if loaded_epoch_num > 0:
                model = _model_loaded.to(device)
                optimizer = _optimizer_loaded
                start_epoch_ssl = loaded_epoch_num
                monitor_ssl.best_metric_val = best_metric_val_resumed
                monitor_ssl.counter = counter_resumed
                monitor_ssl.best_epoch = start_epoch_ssl -1
                current_run_logger.info(f"Resumed SSL pretraining from epoch {start_epoch_ssl}. Monitor state restored.")
            else:
                 current_run_logger.info("No suitable SSL checkpoint loaded. Starting pretraining from scratch.")
                 start_epoch_ssl = 0
        else:
            current_run_logger.info("No SSL checkpoint found by monitor. Starting pretraining from scratch.")
            start_epoch_ssl = 0
    else:
        current_run_logger.info("No SSL checkpoint directory or checkpoints. Starting pretraining from scratch.")
        start_epoch_ssl = 0
    # <<< --- END OF PRETRAIN CHECKPOINT LOADING --- >>>

    current_run_logger.info(f"Starting NMT-EEGPT pretraining loop from epoch {start_epoch_ssl}...")
    pre_training_start_time = time.time()

    for epoch in range(start_epoch_ssl, cfg_ssl.MAX_EPOCHS_PRETRAIN):
        epoch_start_time = time.time()
        model.train()
        running_loss_align = 0.0; running_loss_recon = 0.0; running_total_loss = 0.0
        
        # No zero_grad here if accumulating gradients over multiple batches
        # optimizer.zero_grad() # Moved inside accumulation loop

        for batch_idx, (batch_segment_patches) in enumerate(tqdm(ssl_loader, desc=f"Epoch {epoch+1}/{cfg_ssl.MAX_EPOCHS_PRETRAIN} [SSL]", leave=False, disable=None)):
            batch_segment_patches = batch_segment_patches.to(device) # (B,C,Nt,Pt)
            B, C, N_t, P_t_dims = batch_segment_patches.shape

            patch_mask_flat_for_m_bar = generate_random_masks_for_batch(
                B, C, N_t, cfg_ssl.time_patch_mask_percentage, cfg_ssl.channel_mask_percentage, device)

            # Accumulation requires zeroing grads at the start of an effective batch
            if batch_idx % cfg_ssl.GRAD_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()

            if cfg_ssl.USE_AMP:
                with autocast():
                    pred_align_feats, pred_recon_patches, target_align_feats, target_recon_patches = \
                        model.forward_pretrain(batch_segment_patches, patch_mask_flat_for_m_bar)
                    loss_A = mse_loss(pred_align_feats, F.layer_norm(target_align_feats.detach(), [target_align_feats.size(-1)]))
                    loss_R = mse_loss(pred_recon_patches, F.layer_norm(target_recon_patches.detach(), [target_recon_patches.size(-1)]))
                    total_loss = loss_A + loss_R
                total_loss_scaled = total_loss / cfg_ssl.GRAD_ACCUMULATION_STEPS
                scaler.scale(total_loss_scaled).backward()
            else:
                pred_align_feats, pred_recon_patches, target_align_feats, target_recon_patches = \
                    model.forward_pretrain(batch_segment_patches, patch_mask_flat_for_m_bar)
                loss_A = mse_loss(pred_align_feats, F.layer_norm(target_align_feats.detach(), [target_align_feats.size(-1)]))
                loss_R = mse_loss(pred_recon_patches, F.layer_norm(target_recon_patches.detach(), [target_recon_patches.size(-1)]))
                total_loss = loss_A + loss_R
                total_loss_scaled = total_loss / cfg_ssl.GRAD_ACCUMULATION_STEPS
                total_loss_scaled.backward()

            running_loss_align += loss_A.item()
            running_loss_recon += loss_R.item()
            running_total_loss += total_loss.item()

            if (batch_idx + 1) % cfg_ssl.GRAD_ACCUMULATION_STEPS == 0:
                if cfg_ssl.USE_AMP: scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                
                # Momentum update after optimizer step
                if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                    model.module._update_momentum_encoder()
                else: model._update_momentum_encoder()
                
                # Zero grad for the next accumulation cycle (if not done at the start of the batch)
                # If zero_grad is done at the start of accumulation cycle, this isn't needed here.
                # For safety, if it's at the end of accumulation, this is fine.
                # Let's stick to optimizer.zero_grad() at the start of accumulation steps.

            if scheduler and cfg_ssl.LR_SCHEDULER_PRETRAIN == 'OneCycleLR':
                scheduler.step()
        
        epoch_avg_loss_align = running_loss_align / len(ssl_loader)
        epoch_avg_loss_recon = running_loss_recon / len(ssl_loader)
        epoch_avg_total_loss = running_total_loss / len(ssl_loader)
        epoch_duration = time.time() - epoch_start_time

        metrics_for_ssl_monitor = {
            'total_loss': epoch_avg_total_loss, # This is what monitor_ssl will track
            'align_loss_la': epoch_avg_loss_align,
            'recon_loss_lr': epoch_avg_loss_recon
        }
        
        log_msg_ssl_epoch = (f"Epoch {epoch+1}: Total Loss: {epoch_avg_total_loss:.4f} | Align Loss: {epoch_avg_loss_align:.4f} | "
                             f"Recon Loss: {epoch_avg_loss_recon:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_duration:.2f}s")
        current_run_logger.info(log_msg_ssl_epoch)
        print(log_msg_ssl_epoch)


        if scheduler and cfg_ssl.LR_SCHEDULER_PRETRAIN == 'ReduceLROnPlateau':
            scheduler.step(epoch_avg_total_loss) 
        
        if monitor_ssl.step(epoch, model, optimizer, metrics_for_ssl_monitor, checkpoint_interval=cfg_ssl.PRETRAIN_SAVE_EVERY_EPOCHS):
            current_run_logger.info(f"SSL Pretrain Early stopping decision from monitor at epoch {epoch+1}.")
            break
            
    total_pretraining_time = time.time() - pre_training_start_time
    current_run_logger.info(f"--- NMT-EEGPT Pretraining finished. Total time: {total_pretraining_time // 3600:.0f}h {(total_pretraining_time % 3600) // 60:.0f}m ---")
    if monitor_ssl.best_epoch is not None and monitor_ssl.best_metric_val != float('inf'):
         current_run_logger.info(f"Best SSL model (based on {monitor_ssl.monitor_metric_name}) saved at: {model_save_path_ssl} (Achieved at Epoch: {monitor_ssl.best_epoch+1}, Value: {monitor_ssl.best_metric_val:.4f})")
    else:
         current_run_logger.info(f"No best SSL model saved. Check logs and patience. Last checkpoint is likely the one to use.")


if __name__ == '__main__':
    os.makedirs(cfg_ssl.LOG_DIR_SSL_PRETRAIN, exist_ok=True)
    os.makedirs(cfg_ssl.MODEL_SAVE_DIR_SSL_PRETRAIN, exist_ok=True)
    os.makedirs(cfg_ssl.PREPROCESSED_SSL_DATA_DIR, exist_ok=True)
    
    if not os.path.exists(cfg_ssl.PREPROCESSED_SSL_DATA_DIR) or len(os.listdir(cfg_ssl.PREPROCESSED_SSL_DATA_DIR)) < 10:
        print(f"SSL data directory {cfg_ssl.PREPROCESSED_SSL_DATA_DIR} seems empty or insufficient.")
        print("Please run dataset_utils.py to process NMT EDFs into segments and place them in this directory.")
    else:
        pretrain_nmt_eegpt()