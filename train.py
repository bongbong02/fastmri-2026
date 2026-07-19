import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix
from mraugment import add_mraugment_args


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--model', type=str, default='promptmr', choices=['promptmr', 'varnet'], help='Model architecture')
    parser.add_argument('--cascade', type=int, default=4, help='Number of cascades | 12 in PromptMR+ paper') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='[varnet] Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens_chans', type=int, default=4, help='[varnet] Number of channels for sensitivity map U-Net | 8 in original varnet')

    # PromptMR+ hyperparameters (defaults are a scaled-down config; paper: n_feat0=48,
    # feature_dim 72 96 120, prompt_dim 24 48 72, sens_n_feat0=24, sens_feature_dim 36 48 60,
    # sens_prompt_dim 12 24 36, cascade 12, n_history 11)
    parser.add_argument('--num_adj_slices', type=int, default=5, help='Number of adjacent slices (odd)')
    parser.add_argument('--n_feat0', type=int, default=24, help='Number of channels of first feature extraction')
    parser.add_argument('--feature_dim', type=int, nargs=3, default=[36, 48, 60], help='Feature dims of 3 encoder levels')
    parser.add_argument('--prompt_dim', type=int, nargs=3, default=[12, 24, 36], help='Prompt dims of 3 decoder levels')
    parser.add_argument('--sens_n_feat0', type=int, default=12, help='Sens net: first feature channels')
    parser.add_argument('--sens_feature_dim', type=int, nargs=3, default=[18, 24, 30], help='Sens net: feature dims')
    parser.add_argument('--sens_prompt_dim', type=int, nargs=3, default=[6, 12, 18], help='Sens net: prompt dims')
    parser.add_argument('--len_prompt', type=int, nargs=3, default=[5, 5, 5], help='Number of prompt components per level')
    parser.add_argument('--prompt_size', type=int, nargs=3, default=[64, 32, 16], help='Spatial size of prompts per level')
    parser.add_argument('--n_enc_cab', type=int, nargs=3, default=[2, 3, 3], help='Number of CABs per encoder level')
    parser.add_argument('--n_dec_cab', type=int, nargs=3, default=[2, 2, 3], help='Number of CABs per decoder level')
    parser.add_argument('--n_skip_cab', type=int, nargs=3, default=[1, 1, 1], help='Number of CABs per skip connection')
    parser.add_argument('--n_bottleneck_cab', type=int, default=3, help='Number of CABs in bottleneck')
    parser.add_argument('--n_history', type=int, default=3, help='History features across cascades (PromptMR+; 11 in paper)')
    parser.add_argument('--n_buffer', type=int, default=4, help='Adaptive input buffer channels (PromptMR+)')
    parser.add_argument('--no_adaptive_input', action='store_true', help='Disable adaptive input buffer')
    parser.add_argument('--no_use_ca', action='store_true', help='Disable channel attention')
    parser.add_argument('--learnable_prompt', action='store_true', help='Make prompt parameters learnable')
    parser.add_argument('--no_sens_adj', action='store_true', help='Sens net sees single slices instead of adjacent stack')
    parser.add_argument('--use_checkpoint', action='store_true', help='Gradient checkpointing (slower, less VRAM)')
    parser.add_argument('--compute_sens_per_coil', action='store_true', help='Compute sens maps per coil (slower, less VRAM)')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--grad-clip', type=float, default=0.1, help='Max grad norm (0 disables). Prevents loss blowup on unrolled nets')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps: effective batch = batch_size * grad_accum')
    parser.add_argument('--max-vram-gb', type=float, default=0.0, help='Cap process VRAM to this many GiB (0=off). Fail-fast OOM to verify reproducibility on a smaller GPU')
    # LR schedule: linear warmup (lr_start->lr_peak over lr_warmup_epochs) then
    # cosine decay (lr_peak->lr_final over the remaining epochs). Overrides --lr.
    parser.add_argument('--lr-schedule', action='store_true', help='Enable warmup+cosine LR schedule (overrides constant --lr)')
    parser.add_argument('--lr-warmup-epochs', type=int, default=10, help='Linear warmup length (epochs)')
    parser.add_argument('--lr-start', type=float, default=5e-5, help='LR at epoch 0 (warmup start)')
    parser.add_argument('--lr-peak', type=float, default=2e-4, help='Peak LR at end of warmup')
    parser.add_argument('--lr-final', type=float, default=5e-5, help='Final LR at last epoch (cosine end)')
    parser.add_argument('--resume', action='store_true', help='Resume training from result/<net>/checkpoints/model.pt (weights, optimizer, epoch)')

    # Weights & Biases logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='fastmri-promptmr', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity (team/user)')
    parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name (default: net-name)')

    add_mraugment_args(parser)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
