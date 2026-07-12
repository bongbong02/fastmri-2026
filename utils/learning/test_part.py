import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor
from utils.learning.train_part import build_model

# ---------------------------------------------------------------------------
# Team-editable reconstruction contract.
# recon_eval.py (the fixed timing harness) only calls the three functions
# below. This branch feeds `kspace` + `mask` (k-space domain) to a PromptMR+
# model (or VarNet fallback); the model is rebuilt from the hyperparameters
# stored in the checkpoint, so the harness CLI args stay untouched.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"      # harness delivers the kspace H5 to prep_volume


def load_model(args, device):
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    train_args = checkpoint.get('args', args)
    model = build_model(train_args).to(device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # data loading must match the trained slice neighborhood
    model.inference_num_adj_slices = getattr(train_args, 'num_adj_slices', 1)
    return model


def prep_volume(image_path, kspace_path, device):
    """Load one volume's k-space and mask onto the host. Untimed: no model compute here."""
    with h5py.File(kspace_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask = np.array(hf['mask'])
    return {"kspace": kspace, "mask": mask, "device": device, "num_slices": kspace.shape[0]}


def _gather_adj_slices(kspace_vol, s, num_adj_slices):
    """Stack the clamped [s-k .. s+k] neighborhood along the coil axis."""
    if num_adj_slices == 1:
        return kspace_vol[s]
    half = num_adj_slices // 2
    n = kspace_vol.shape[0]
    idx = [min(max(s + i, 0), n - 1) for i in range(-half, half + 1)]
    return np.concatenate([kspace_vol[j] for j in idx], axis=0)


def recon_slice(model, ctx, s):
    """Reconstruct a single slice (batch=1). Timed by the harness."""
    device = ctx["device"]
    mask = ctx["mask"]
    num_adj_slices = getattr(model, 'inference_num_adj_slices', getattr(model, 'num_adj_slices', 1))
    kspace_np = _gather_adj_slices(ctx["kspace"], s, num_adj_slices)
    kspace = to_tensor(kspace_np * mask)
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0).to(device=device)
    mask_t = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
    mask_t = mask_t.unsqueeze(0).to(device=device)
    return model(kspace, mask_t)[0]


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = load_model(args, device)
    args.num_adj_slices = model.inference_num_adj_slices

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
