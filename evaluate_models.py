# NMT_EEGPT_Project/evaluate_models.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import argparse
import json # For saving metrics dictionary

# Import configurations, datasets, models, and monitors
import config_supervised as cfg_sup
import config_ssl_finetune as cfg_ft
import config_ssl_pretrain as cfg_ssl_pretrain_model_params # For NMT-EEGPT structure

from dataset_supervised import SupervisedNMTDataset # Test set will be supervised
from models.hybrid_cnn_transformer import HybridCNNTransformer
from models.ctnet_model import CTNet
from models.eeg_conformer_model import EEGConformer
from models.nmt_eegpt_downstream_model import NMT_EEGPT_Classifier # For loading NMT-EEGPT
from monitors import PerformanceMetricsMonitor

EVAL_LOG_DIR = 'logs/evaluation_results/'
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

def setup_eval_logging(log_dir, model_name_str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'evaluation_log_{model_name_str}.txt')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler) # Clear existing to avoid duplicates if any
    return log_file

def evaluate_single_model(model_type, model_path_to_load):
    log_file = setup_eval_logging(EVAL_LOG_DIR, model_type)
    logging.info(f"--- Starting Evaluation for: {model_type} from {model_path_to_load} ---")
    
    if not os.path.exists(model_path_to_load):
        logging.error(f"Model path not found: {model_path_to_load}. Exiting.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Data Loading (NMT 'eval' split) ---
    # Using parameters from config_supervised as it defines the main dataset properties
    logging.info("Loading EVALUATION dataset (NMT 'eval' split)...")
    eval_dataset = SupervisedNMTDataset(
        data_dir=cfg_sup.PROCESSED_DATA_SUPERVISED_DIR, # Processed segments for supervised
        split_type='eval', # Crucially, use the 'eval' split
        augment=False,
        segment_duration_sec=cfg_sup.SEGMENT_DURATION_SEC,
        target_sfreq=cfg_sup.TARGET_SFREQ,
        test_mode_reduce_data=False # Evaluate on full eval set
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg_sup.BATCH_SIZE, # Can use supervised batch size
        shuffle=False, num_workers=4, pin_memory=True
    )
    logging.info(f"Evaluation segments: {len(eval_dataset)}")
    if len(eval_dataset) == 0:
        logging.error("Evaluation dataset is empty. Ensure dataset_utils.py has run and `eval` split exists.")
        return

    # --- Model Initialization & Loading Weights ---
    model = None
    if model_type == 'HybridCNNTransformer':
        model = HybridCNNTransformer( # Params from config_supervised
            n_channels=cfg_sup.N_CHANNELS_SELECTED, n_start_chans=cfg_sup.HYBRID_N_START_CHANS,
            n_layers_transformer=cfg_sup.HYBRID_N_LAYERS_TRANSFORMER, n_heads=cfg_sup.HYBRID_N_HEADS,
            hidden_dim=cfg_sup.HYBRID_HIDDEN_DIM, ff_dim=cfg_sup.HYBRID_FF_DIM,
            dropout=cfg_sup.HYBRID_DROPOUT, input_time_length=cfg_sup.INPUT_TIME_LENGTH,
            n_classes=cfg_sup.N_CLASSES
        )
    elif model_type == 'CTNet':
        model = CTNet( # Params from config_supervised
            n_channels=cfg_sup.N_CHANNELS_SELECTED, n_classes=cfg_sup.N_CLASSES,
            input_time_length=cfg_sup.INPUT_TIME_LENGTH, target_sfreq=cfg_sup.TARGET_SFREQ,
            f1=cfg_sup.CTNET_F1, d_multiplier=cfg_sup.CTNET_D, f2=cfg_sup.CTNET_F2,
            kc1_divisor=4, pool1_size=cfg_sup.CTNET_P1, k2_kernel_length=cfg_sup.CTNET_K2, 
            pool2_size=cfg_sup.CTNET_P2, transformer_depth=cfg_sup.CTNET_TRANSFORMER_DEPTH, 
            transformer_heads=cfg_sup.CTNET_TRANSFORMER_HEADS, dropout_cnn=cfg_sup.CTNET_DROPOUT_CNN, 
            dropout_transformer_p1=cfg_sup.CTNET_DROPOUT_TRANSFORMER, 
            dropout_classifier_p2=cfg_sup.CTNET_DROPOUT_CLASSIFIER
        )
    elif model_type == 'EEGConformer':
        model = EEGConformer( # Params from config_supervised
            n_channels=cfg_sup.N_CHANNELS_SELECTED, n_classes=cfg_sup.N_CLASSES,
            input_time_length=cfg_sup.INPUT_TIME_LENGTH, target_sfreq=cfg_sup.TARGET_SFREQ,
            n_filters_time=cfg_sup.CONFORMER_N_FILTERS_TIME, filter_time_length_ms=100,
            n_filters_spat=cfg_sup.CONFORMER_N_FILTERS_SPAT, pool_time_length_ms=300,
            pool_time_stride_ms=60, cnn_drop_prob=cfg_sup.CONFORMER_DROPOUT,
            transformer_d_model=None, transformer_depth=cfg_sup.CONFORMER_TRANSFORMER_DEPTH,
            transformer_n_heads=cfg_sup.CONFORMER_TRANSFORMER_HEADS,
            transformer_drop_prob=cfg_sup.CONFORMER_DROPOUT, classifier_hidden_dim=128,
            classifier_drop_prob=cfg_sup.CONFORMER_DROPOUT
        )
    elif model_type == 'NMT_EEGPT_Classifier':
        # Load structure params from SSL pretrain config, downstream params from finetune config
        model = NMT_EEGPT_Classifier(
            n_channels_model=cfg_ssl_pretrain_model_params.N_CHANNELS_MODEL,
            segment_time_len_samples=cfg_ssl_pretrain_model_params.INPUT_TIME_LENGTH_MODEL,
            patch_time_len_samples=cfg_ssl_pretrain_model_params.PATCH_TIME_LENGTH_SAMPLES,
            embed_dim=cfg_ssl_pretrain_model_params.EMBED_DIM,
            encoder_layers=cfg_ssl_pretrain_model_params.ENCODER_LAYERS,
            num_heads=cfg_ssl_pretrain_model_params.NUM_HEADS,
            ff_dim=cfg_ssl_pretrain_model_params.FEEDFORWARD_DIM,
            dropout_transformer=cfg_ssl_pretrain_model_params.DROPOUT_PRETRAIN,
            num_summary_tokens=cfg_ssl_pretrain_model_params.NUM_SUMMARY_TOKENS,
            n_classes=cfg_ft.N_CLASSES, # Downstream classes
            use_adaptive_spatial_filter=cfg_ft.USE_ADAPTIVE_SPATIAL_FILTER,
            n_input_channels_to_asf=cfg_ft.N_CHANNELS_INPUT_TO_MODEL, # e.g. 21
            pretrained_encoder_path=None, # Don't need to load here, will load full model state dict
            freeze_encoder=False # Does not matter for eval, as we load full trained model
        )
    else:
        logging.error(f"Unknown model_type: {model_type}")
        return

    try:
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        logging.info(f"Successfully loaded model weights from {model_path_to_load}")
    except Exception as e:
        logging.error(f"Error loading model weights from {model_path_to_load}: {e}", exc_info=True)
        return
        
    model.to(device)
    model.eval()

    # --- Evaluate ---
    logging.info("Evaluating model on the test set...")
    # For loss calculation during evaluation (optional)
    # criterion_eval = nn.CrossEntropyLoss() # Or FocalLoss if that's what was used in training for fair comparison
    
    eval_monitor = PerformanceMetricsMonitor(model, eval_loader, device, criterion=None) # No criterion for pure eval
    metrics = eval_monitor.evaluate()

    # --- Log and Save Metrics ---
    logging.info(f"--- Evaluation Metrics for {model_type} ---")
    for key, value in metrics.items():
        if key not in ['predictions_list', 'labels_list', 'probabilities_class1_list']:
            logging.info(f"  {key}: {value}")
            print(f"  {key}: {value}")
        elif key == 'confusion_matrix':
             logging.info(f"  Confusion Matrix:\n{value}")
             print(f"  Confusion Matrix:\n{value}")


    # Save metrics dictionary and raw predictions/labels/probs
    metrics_save_path = os.path.join(EVAL_LOG_DIR, f'evaluation_metrics_{model_type}.json')
    # Convert numpy arrays in metrics to lists for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        else:
            metrics_serializable[k] = v
            
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    logging.info(f"Saved detailed metrics to {metrics_save_path}")

    # For plotting later, visualize_results.py can load these .npy files
    np.save(os.path.join(EVAL_LOG_DIR, f'predictions_{model_type}.npy'), metrics['predictions_list'])
    np.save(os.path.join(EVAL_LOG_DIR, f'labels_{model_type}.npy'), metrics['labels_list'])
    np.save(os.path.join(EVAL_LOG_DIR, f'probabilities_{model_type}.npy'), metrics['probabilities_class1_list'])
    logging.info(f"Saved predictions, labels, and probabilities for {model_type} in {EVAL_LOG_DIR}")

    logging.info(f"--- Evaluation complete for {model_type} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained EEG classification models.")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['HybridCNNTransformer', 'CTNet', 'EEGConformer', 'NMT_EEGPT_Classifier'],
                        help="Type of the model to evaluate.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model weights (.pt file).")
    args = parser.parse_args()

    evaluate_single_model(args.model_type, args.model_path)

    # Example usage:
    # python evaluate_models.py --model_type HybridCNNTransformer --model_path models/saved_supervised/HybridCNNTransformer_best.pt
    # python evaluate_models.py --model_type NMT_EEGPT_Classifier --model_path models/saved_ssl_finetune/nmt_eegpt_downstream_linear_probe_best.pt