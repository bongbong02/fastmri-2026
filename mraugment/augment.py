"""A small, self-contained MRAugment implementation.

The pipeline follows Fabian et al. (ICML 2021): transform the complex coil
images and then recreate both full k-space and the RSS target.  This preserves
the measurement/target relationship and the k-space noise statistics.

Adapted from https://github.com/AIF4S/MRAugment (MIT, Zalan Fabian, 2021).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _ifft2c(kspace):
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1))


def _fft2c(image):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(kspace, dim=(-2, -1))


def _center_crop(x, shape):
    h, w = x.shape[-2:]
    out_h, out_w = min(int(shape[0]), h), min(int(shape[1]), w)
    top, left = (h - out_h) // 2, (w - out_w) // 2
    return x[..., top : top + out_h, left : left + out_w]


def _center_fit(x, shape):
    """Center crop/pad trailing dimensions to ``shape``."""
    out_h, out_w = map(int, shape)
    x = _center_crop(x, shape)
    h, w = x.shape[-2:]
    dh, dw = out_h - h, out_w - w
    if dh or dw:
        x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
    return x


def _reflect_translate(x, dy, dx):
    h, w = x.shape[-2:]
    dy, dx = max(-h + 1, min(h - 1, dy)), max(-w + 1, min(w - 1, dx))
    pad = (max(dx, 0), max(-dx, 0), max(-dy, 0), max(dy, 0))
    x = F.pad(x, pad, mode="reflect")
    top = max(dy, 0)
    left = max(-dx, 0)
    return x[..., top : top + h, left : left + w]


class MRAugment:
    """Apply matched spatial transforms to complex coil images and RSS target."""

    names = ("translation", "rotation", "scaling", "shearing", "rot90", "fliph", "flipv")

    def __init__(self, args):
        self.enabled = bool(args.mraugment)
        # coils are stacked as [slice0_coils, slice1_coils, ...] along dim 0; the
        # target must be the RSS of the center slice's coils only.
        self.num_adj_slices = getattr(args, "num_adj_slices", 1)
        self.schedule = args.aug_schedule
        self.delay = args.aug_delay
        self.strength = args.aug_strength
        self.exp_decay = args.aug_exp_decay
        self.epochs = args.num_epochs
        self.epoch = 0
        self.rng = np.random.RandomState(args.seed)
        self.weights = {name: getattr(args, f"aug_weight_{name}") for name in self.names}
        self.max_translation_x = args.aug_max_translation_x
        self.max_translation_y = args.aug_max_translation_y
        self.max_rotation = args.aug_max_rotation
        self.max_shearing_x = args.aug_max_shearing_x
        self.max_shearing_y = args.aug_max_shearing_y
        self.max_scaling = args.aug_max_scaling

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def probability(self):
        if not self.enabled or self.epoch < self.delay:
            return 0.0
        # Epochs are zero-indexed. Including the current epoch makes one-epoch
        # smoke runs useful and reaches exactly p_max in the final epoch.
        duration = max(self.epochs - self.delay, 1)
        progress = min(max((self.epoch - self.delay + 1) / duration, 0.0), 1.0)
        if self.schedule == "constant":
            factor = 1.0
        elif self.schedule == "ramp":
            factor = progress
        else:
            factor = (1.0 - math.exp(-self.exp_decay * progress)) / (1.0 - math.exp(-self.exp_decay))
        return min(max(self.strength * factor, 0.0), 1.0)

    def _apply(self, name, p):
        return self.rng.uniform() < min(1.0, p * self.weights[name])

    def __call__(self, kspace, target_shape):
        """Return augmented complex k-space and its matched float RSS target."""
        p = self.probability()
        if p <= 0:
            return kspace, None

        original_shape = kspace.shape[-2:]
        image = _ifft2c(kspace)
        # grid_sample operates on real channels; every coil/component receives
        # exactly the same spatial transform.
        x = torch.view_as_real(image).permute(0, 3, 1, 2).reshape(1, -1, *original_shape)

        if self._apply("fliph", p):
            x = torch.flip(x, (-1,))
        if self._apply("flipv", p):
            x = torch.flip(x, (-2,))
        if self._apply("rot90", p):
            x = torch.rot90(x, int(self.rng.randint(1, 4)), (-2, -1))
            x = _center_fit(x, original_shape)
        if self._apply("translation", p):
            dy = int(self.rng.uniform(-self.max_translation_y, self.max_translation_y) * original_shape[0])
            dx = int(self.rng.uniform(-self.max_translation_x, self.max_translation_x) * original_shape[1])
            x = _reflect_translate(x, dy, dx)

        angle = self.rng.uniform(-self.max_rotation, self.max_rotation) if self._apply("rotation", p) else 0.0
        shear_x = self.rng.uniform(-self.max_shearing_x, self.max_shearing_x) if self._apply("shearing", p) else 0.0
        shear_y = self.rng.uniform(-self.max_shearing_y, self.max_shearing_y) if shear_x else 0.0
        scale = self.rng.uniform(1 - self.max_scaling, 1 + self.max_scaling) if self._apply("scaling", p) else 1.0
        if angle or shear_x or shear_y or scale != 1.0:
            a, sx, sy = map(math.radians, (angle, shear_x, shear_y))
            rotation = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=x.dtype)
            shear = torch.tensor([[1.0, math.tan(sx)], [math.tan(sy), 1.0]], dtype=x.dtype)
            matrix = (rotation @ shear) / scale
            theta = torch.zeros((1, 2, 3), dtype=x.dtype)
            theta[0, :, :2] = matrix
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        x = x.reshape(-1, 2, *original_shape).permute(0, 2, 3, 1).contiguous()
        image = torch.view_as_complex(x)
        # target = RSS of the center slice's coils only, center crop/pad to
        # target_shape so it matches the model output (utils.common.center_crop
        # zero-pads narrow k-space up to 384; plain crop would leave it 384x372).
        coils = image.shape[0] // self.num_adj_slices
        center = self.num_adj_slices // 2
        center_coils = image[center * coils:(center + 1) * coils]
        target = _center_fit(torch.sqrt(torch.sum(torch.abs(center_coils) ** 2, dim=0)), target_shape).float()
        return _fft2c(image), target


def add_mraugment_args(parser):
    group = parser.add_argument_group("MRAugment")
    group.add_argument("--mraugment", action="store_true", help="Enable physics-aware training augmentation")
    group.add_argument("--aug-schedule", choices=("constant", "ramp", "exp"), default="exp")
    group.add_argument("--aug-delay", type=int, default=0)
    group.add_argument("--aug-strength", type=float, default=0.55)
    group.add_argument("--aug-exp-decay", type=float, default=5.0)
    for name, default in (("translation", 1.0), ("rotation", 0.5), ("scaling", 1.0),
                          ("shearing", 1.0), ("rot90", 0.5), ("fliph", 0.5), ("flipv", 0.5)):
        group.add_argument(f"--aug-weight-{name}", type=float, default=default)
    group.add_argument("--aug-max-translation-x", type=float, default=0.10)
    group.add_argument("--aug-max-translation-y", type=float, default=0.08)
    group.add_argument("--aug-max-rotation", type=float, default=180.0)
    group.add_argument("--aug-max-shearing-x", type=float, default=15.0)
    group.add_argument("--aug-max-shearing-y", type=float, default=15.0)
    group.add_argument("--aug-max-scaling", type=float, default=0.25)
    return parser
