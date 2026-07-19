import shutil
import math
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet
from mraugment import MRAugment
from utils.model.promptmr import PromptMR

import os


def lr_at_epoch(args, epoch):
    """Linear warmup (lr_start -> lr_peak over lr_warmup_epochs) then cosine decay
    (lr_peak -> lr_final over the remaining epochs). Pure function of epoch, so it
    reproduces exactly on --resume."""
    warm = max(1, args.lr_warmup_epochs)
    total = args.num_epochs
    start, peak, final = args.lr_start, args.lr_peak, args.lr_final
    if epoch < warm:
        return start + (peak - start) * (epoch / warm)
    p = (epoch - warm) / max(1, total - warm)
    p = min(max(p, 0.0), 1.0)
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * p))


def build_model(args):
    if getattr(args, 'model', 'varnet') == 'promptmr':
        return PromptMR(
            num_cascades=args.cascade,
            num_adj_slices=args.num_adj_slices,
            n_feat0=args.n_feat0,
            feature_dim=list(args.feature_dim),
            prompt_dim=list(args.prompt_dim),
            sens_n_feat0=args.sens_n_feat0,
            sens_feature_dim=list(args.sens_feature_dim),
            sens_prompt_dim=list(args.sens_prompt_dim),
            len_prompt=list(args.len_prompt),
            prompt_size=list(args.prompt_size),
            n_enc_cab=list(args.n_enc_cab),
            n_dec_cab=list(args.n_dec_cab),
            n_skip_cab=list(args.n_skip_cab),
            n_bottleneck_cab=args.n_bottleneck_cab,
            no_use_ca=args.no_use_ca,
            learnable_prompt=args.learnable_prompt,
            adaptive_input=not args.no_adaptive_input,
            n_buffer=args.n_buffer,
            n_history=args.n_history,
            use_sens_adj=not args.no_sens_adj,
            use_checkpoint=args.use_checkpoint,
            compute_sens_per_coil=args.compute_sens_per_coil,
        )
    return VarNet(num_cascades=args.cascade,
                  chans=args.chans,
                  sens_chans=args.sens_chans)


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, run=None):
    model.train()
    data_loader.dataset.transform.set_epoch(epoch)
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    grad_clip = getattr(args, 'grad_clip', 0.0)
    accum = max(1, getattr(args, 'grad_accum', 1))
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        # scale by 1/accum so the accumulated grad is the mean over the window
        # (matches a real batch of size batch_size*accum).
        (loss / accum).backward()
        if torch.isfinite(loss):
            total_loss += loss.item()

        # step only at the end of an accumulation window (or the last iter).
        is_step = ((iter + 1) % accum == 0) or (iter + 1 == len_loader)
        grad_norm = None
        if is_step:
            # clip gradients: unrolled nets + SSIM loss occasionally produce
            # large-norm gradients that otherwise blow up the weights.
            if grad_clip and grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            # Non-finite guard: one unstable micro-batch yields a nan/inf grad.
            # clip_grad_norm_ does NOT stop this -- total_norm becomes nan and the
            # clip coefficient (max_norm/nan) multiplies every grad by nan,
            # poisoning ALL weights permanently (loss -> nan forever). Discard the
            # whole window instead so training rides through. Logged to stay visible.
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f'[skip] non-finite grad at epoch {epoch} iter {iter}: '
                      f'grad_norm={float(grad_norm):.4g}')
                if run is not None:
                    run.log({'train/skipped': 1, 'iter': epoch * len_loader + iter})
            optimizer.zero_grad()

        if run is not None:
            log = {'train/loss': loss.item(), 'iter': epoch * len_loader + iter}
            if grad_norm is not None:
                log['train/grad_norm'] = float(grad_norm)
            run.log(log)

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def _wandb_init(args):
    """Return the wandb run if --wandb is set and the package is importable, else None."""
    if not getattr(args, 'wandb', False):
        return None
    try:
        import wandb
    except ImportError:
        print('[wandb] package not installed; skipping logging')
        return None
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or str(args.net_name),
        config=config,
    )


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # Cap the process to a smaller VRAM budget so an over-budget config OOMs at
    # iter 0 instead of after hours. Lets an 8GB target be verified on a 16GB node.
    max_vram = getattr(args, 'max_vram_gb', 0.0)
    if max_vram and max_vram > 0 and torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        frac = min(1.0, max_vram / total_gib)
        torch.cuda.set_per_process_memory_fraction(frac, device)
        print(f'VRAM cap: {max_vram:.1f} GiB of {total_gib:.1f} GiB total (fraction {frac:.3f})')

    run = _wandb_init(args)

    model = build_model(args)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    # resume from the last saved epoch (model.pt) if requested
    if getattr(args, 'resume', False):
        ckpt_path = args.exp_dir / 'model.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # move optimizer state to the training device (loaded on CPU above)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = ckpt['epoch']  # saved as epoch+1, i.e. next epoch to run
            best_val_loss = float(ckpt['best_val_loss'])
            print(f'Resumed from {ckpt_path}: start_epoch={start_epoch}, best_val_loss={best_val_loss:.4g}')
        else:
            print(f'--resume set but {ckpt_path} not found; starting from scratch')


    augmentor = MRAugment(args) if args.mraugment else None
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True, augmentor=augmentor)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        if getattr(args, 'lr_schedule', False):
            cur_lr = lr_at_epoch(args, epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr
            print(f'LR = {cur_lr:.3e} (epoch {epoch})')
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, run=run)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if run is not None:
            run.log({
                'epoch': epoch,
                'train/loss_epoch': float(train_loss),
                'val/loss': float(val_loss),
                'val/best_loss': float(best_val_loss),
                'lr': cur_lr,
                'train_time_s': train_time,
                'val_time_s': val_time,
                'aug_prob': augmentor.probability() if augmentor is not None else 0.0,
            })

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

    if run is not None:
        run.finish()
