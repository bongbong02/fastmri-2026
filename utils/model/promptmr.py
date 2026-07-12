"""
PromptMR+ model (promptmr_v2 from https://github.com/hellopipu/PromptMR-plus),
adapted to this skeleton:
  - einops replaced by view/permute
  - vendored fastmri ops
  - forward(masked_kspace, mask) returns the center-cropped 384x384 magnitude
    image of the central slice, matching the VarNet contract of this repo
"""
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

import fastmri
from utils.common.utils import center_crop
from promptmr_utils import (
    KspaceACSExtractor, conv, CAB, DownBlock, UpBlock, SkipBlock, PromptBlock,
    sens_expand, sens_reduce,
)


class PromptUnet(nn.Module):
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 n_feat0: int,
                 feature_dim: List[int],
                 prompt_dim: List[int],
                 len_prompt: List[int],
                 prompt_size: List[int],
                 n_enc_cab: List[int],
                 n_dec_cab: List[int],
                 n_skip_cab: List[int],
                 n_bottleneck_cab: int,
                 kernel_size=3,
                 reduction=4,
                 act=nn.PReLU(),
                 bias=False,
                 no_use_ca=False,
                 learnable_prompt=False,
                 adaptive_input=False,
                 n_buffer=0,
                 n_history=0,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0

        in_chans = in_chans * (1 + self.n_buffer) if adaptive_input else in_chans
        out_chans = out_chans * (1 + self.n_buffer) if adaptive_input else in_chans

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act, no_use_ca)
                                          for _ in range(n_bottleneck_cab)])
        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x, history_feat: Optional[List[torch.Tensor]] = None):
        if history_feat is None:
            history_feat = [None, None, None]

        history_feat3, history_feat2, history_feat1 = history_feat
        current_feat = []

        # 0. feature extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        current_feat.append(x.clone())
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

        # 4. update history features for the next cascade
        if self.n_history > 0:
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:  # for the first cascade, repeat the current feature
                    history_feat[i] = torch.cat([torch.tile(current_feat[i], (1, self.n_history, 1, 1))], dim=1)
                else:  # for the rest cascades: pop the oldest feature and append the current feature
                    history_feat[i] = torch.cat([current_feat[i], history_feat[i][:, :-self.feature_dim[2 - i]]], dim=1)
        return self.conv_last(x), history_feat


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        learnable_prompt=False,
        adaptive_input=False,
        n_buffer=0,
        n_history=0,
    ):
        super().__init__()
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.unet = PromptUnet(in_chans=in_chans,
                               out_chans=out_chans,
                               n_feat0=n_feat0,
                               feature_dim=feature_dim,
                               prompt_dim=prompt_dim,
                               len_prompt=len_prompt,
                               prompt_size=prompt_size,
                               n_enc_cab=n_enc_cab,
                               n_dec_cab=n_dec_cab,
                               n_skip_cab=n_skip_cab,
                               n_bottleneck_cab=n_bottleneck_cab,
                               no_use_ca=no_use_ca,
                               learnable_prompt=learnable_prompt,
                               adaptive_input=adaptive_input,
                               n_buffer=n_buffer,
                               n_history=n_history,
                               )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None):
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[1]
        if buffer is not None:
            x = torch.cat([x, buffer], dim=1)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x, history_feat = self.unet(x, history_feat)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if buffer is not None:
            x, _, latent, _ = torch.split(x, [cc, cc, cc, x.shape[1] - 3 * cc], dim=1)
        else:
            latent = None
        return x, latent, history_feat


class PromptMRBlock(nn.Module):
    def __init__(self, model: nn.Module, num_adj_slices=5):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.n_buffer = model.n_buffer
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_img)
        current_kspace = sens_expand(current_img, sens_maps, self.num_adj_slices)
        ffx = sens_reduce(torch.where(mask.bool(), current_kspace, zero), sens_maps, self.num_adj_slices)
        if self.n_buffer > 0:
            # adaptive input. buffer: A^H*A*x_i, latent, x0, A^H*A*x_i - x0
            buffer = torch.cat([ffx] + [latent] * (self.n_buffer - 3) + [img_zf, ffx - img_zf], dim=1)
        else:
            buffer = None

        soft_dc = (ffx - img_zf) * self.dc_weight
        model_term, latent, history_feat = self.model(current_img, history_feat, buffer)
        img_pred = current_img - soft_dc - model_term
        return img_pred, latent, history_feat


class SensitivityModel(nn.Module):
    def __init__(
        self,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        mask_center: bool = True,
        learnable_prompt=False,
        use_sens_adj: bool = True,
    ):
        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.use_sens_adj = use_sens_adj
        self.norm_unet = NormPromptUnet(in_chans=2 * self.num_adj_slices if use_sens_adj else 2,
                                        out_chans=2 * self.num_adj_slices if use_sens_adj else 2,
                                        n_feat0=n_feat0,
                                        feature_dim=feature_dim,
                                        prompt_dim=prompt_dim,
                                        len_prompt=len_prompt,
                                        prompt_size=prompt_size,
                                        n_enc_cab=n_enc_cab,
                                        n_dec_cab=n_dec_cab,
                                        n_skip_cab=n_skip_cab,
                                        n_bottleneck_cab=n_bottleneck_cab,
                                        no_use_ca=no_use_ca,
                                        learnable_prompt=learnable_prompt,
                                        )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, adj_coil, h, w, two = x.shape
        if self.use_sens_adj:
            # b (adj coil) h w two -> (b coil) adj h w two
            coil = adj_coil // self.num_adj_slices
            x = x.view(b, self.num_adj_slices, coil, h, w, two)
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(b * coil, self.num_adj_slices, h, w, two)
        else:
            x = x.reshape(b * adj_coil, 1, h, w, two)
        return x, b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, adj, h, w, two = x.shape
        if self.use_sens_adj:
            # (b coil) adj h w two -> b (adj coil) h w two
            coil = bc // batch_size
            x = x.view(batch_size, coil, adj, h, w, two)
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(batch_size, adj * coil, h, w, two)
        else:
            x = x.view(batch_size, bc // batch_size, h, w, two)
        return x

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil // self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / fastmri.rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        bc = images.shape[0]  # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0))[0])
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[torch.Tensor] = None,
        compute_per_coil: bool = False,
    ) -> torch.Tensor:
        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies)
        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace_acs))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet, images, compute_per_coil), batches)
        )


class PromptMR(nn.Module):
    """
    PromptMR+ unrolled model. forward(masked_kspace, mask) keeps the same
    contract as this repo's VarNet: input (b, adj*coil, h, w, 2) k-space +
    (b, 1, 1, w, 1) mask, output (b, 384, 384) magnitude image.
    """

    def __init__(
        self,
        num_cascades: int = 4,
        num_adj_slices: int = 5,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        sens_n_feat0: int = 12,
        sens_feature_dim: List[int] = [18, 24, 30],
        sens_prompt_dim: List[int] = [6, 12, 18],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[int] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
    ):
        super().__init__()
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca=sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            learnable_prompt=learnable_prompt,
            use_sens_adj=use_sens_adj,
        )
        # DC + denoiser in each cascade
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    no_use_ca=no_use_ca,
                    learnable_prompt=learnable_prompt,
                    adaptive_input=adaptive_input,
                    n_buffer=n_buffer,
                    n_history=n_history,
                ),
                num_adj_slices=num_adj_slices,
            ) for _ in range(num_cascades)
        ])

    def _reconstruct(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the unrolled network and return the uncropped rss magnitude of the
        central slice, shape (b, h, w)."""
        use_checkpoint = self.use_checkpoint and self.training and torch.is_grad_enabled()

        if use_checkpoint:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, None, self.compute_sens_per_coil,
                use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, None, self.compute_sens_per_coil)

        img_zf = sens_reduce(masked_kspace, sens_maps, self.num_adj_slices)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None

        for cascade in self.cascades:
            if use_checkpoint:
                img_pred, latent, history_feat = torch.utils.checkpoint.checkpoint(
                    cascade, img_pred, img_zf, latent, mask, sens_maps, history_feat, use_reentrant=False)
            else:
                img_pred, latent, history_feat = cascade(img_pred, img_zf, latent, mask, sens_maps, history_feat)

        # combine central slice with its sens maps, rss magnitude
        img_pred = torch.chunk(img_pred, self.num_adj_slices, dim=1)[self.center_slice]
        sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        return fastmri.rss(fastmri.complex_abs(fastmri.complex_mul(img_pred, sens_maps)), dim=1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_kspace: (b, num_adj_slices*coil, h, w, 2) input k-space
            mask: (b, 1, 1, w, 1) undersampling mask
        Returns:
            (b, 384, 384) magnitude image of the central slice
        """
        result = self._reconstruct(masked_kspace, mask)
        return center_crop(result, 384, 384)
