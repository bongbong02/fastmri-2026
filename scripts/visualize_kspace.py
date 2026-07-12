#!/usr/bin/env python3
"""Visualizer for FastMRI challenge HDF5 data.

Handles two file types found under data/Data/{train,val}/:

  kspace/*.h5  -> kspace (slices, coils, H, W) complex64, mask (W,)
  image/*.h5   -> image_grappa / image_input / image_label (slices, H, W) float32,
                  plus optional 'annotations' attribute (bounding boxes)

Usage:
  python visualize_kspace.py PATH.h5                 # auto-detect type, middle slice
  python visualize_kspace.py PATH.h5 --slice 13
  python visualize_kspace.py PATH.h5 --coil 3        # kspace: single coil instead of RSS
  python visualize_kspace.py PATH.h5 --save out.png  # save instead of showing
  python visualize_kspace.py PATH.h5 --boxes-all     # image: boxes on all 3 panels

Annotation boxes (bounding boxes from the file-level 'annotations' attribute) are
drawn on the image_label panel automatically. When no --slice is given for an image
file, the first annotated slice is shown so the boxes are visible by default.

If the file is a kspace file, the script shows the raw k-space (log magnitude),
per-coil zero-filled reconstructions, and the root-sum-of-squares combined image.
"""
import argparse
import json
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# reconstruction helpers
# ---------------------------------------------------------------------------
def ifft2c(kspace):
    """Centered 2D inverse FFT over the last two axes."""
    axes = (-2, -1)
    x = np.fft.ifftshift(kspace, axes=axes)
    x = np.fft.ifft2(x, axes=axes)
    x = np.fft.fftshift(x, axes=axes)
    return x


def rss(coil_images, axis=0):
    """Root-sum-of-squares coil combination."""
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=axis))


def log_mag(k, eps=1e-9):
    """Log magnitude of complex k-space for display."""
    return np.log(np.abs(k) + eps)


# ---------------------------------------------------------------------------
# file type detection
# ---------------------------------------------------------------------------
def detect_type(f):
    if "kspace" in f:
        return "kspace"
    if any(k in f for k in ("image_label", "image_input", "image_grappa")):
        return "image"
    raise ValueError(f"Unknown file layout, keys = {list(f.keys())}")


# ---------------------------------------------------------------------------
# kspace visualization
# ---------------------------------------------------------------------------
def show_kspace(f, args):
    kspace = f["kspace"]  # (slices, coils, H, W)
    n_slices, n_coils = kspace.shape[0], kspace.shape[1]
    sl = args.slice if args.slice is not None else n_slices // 2
    sl = int(np.clip(sl, 0, n_slices - 1))

    ks = np.asarray(kspace[sl])  # (coils, H, W)
    coil_imgs = ifft2c(ks)       # (coils, H, W) complex
    recon_rss = rss(coil_imgs, axis=0)

    mask = np.asarray(f["mask"]) if "mask" in f else None

    if args.coil is not None:
        c = int(np.clip(args.coil, 0, n_coils - 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(log_mag(ks[c]), cmap="gray")
        axes[0].set_title(f"k-space (log mag) coil {c}")
        axes[1].imshow(np.abs(coil_imgs[c]), cmap="gray")
        axes[1].set_title(f"coil {c} image |IFFT|")
        axes[2].imshow(recon_rss, cmap="gray")
        axes[2].set_title("RSS combined")
        for a in axes:
            a.axis("off")
    else:
        # grid: full-coil RSS k-space, RSS recon, mask, and a montage of coils
        ncols = 4
        fig, axes = plt.subplots(1, ncols, figsize=(20, 5))

        ks_rss = rss(ks, axis=0)  # combine coils in k-space for a single view
        axes[0].imshow(log_mag(ks_rss), cmap="gray")
        axes[0].set_title("k-space RSS (log mag)")

        axes[1].imshow(recon_rss, cmap="gray")
        axes[1].set_title("RSS reconstruction")

        # coil montage (magnitude images)
        montage = _montage(np.abs(coil_imgs))
        axes[2].imshow(montage, cmap="gray")
        axes[2].set_title(f"{n_coils} coil images")

        if mask is not None:
            # broadcast 1D sampling mask to 2D for display
            m2d = np.tile(mask.reshape(1, -1), (min(ks.shape[1], 200), 1))
            axes[3].imshow(m2d, cmap="gray", aspect="auto")
            axes[3].set_title(f"mask ({int(mask.sum())}/{mask.size} lines)")
        else:
            axes[3].imshow(recon_rss, cmap="gray")
            axes[3].set_title("RSS reconstruction")

        for a in axes:
            a.axis("off")

    fig.suptitle(f"{os.path.basename(args.path)}  slice {sl}/{n_slices - 1}  "
                 f"coils={n_coils}  shape={tuple(kspace.shape)}")
    fig.tight_layout()
    return fig


def _montage(vol, pad=2):
    """Arrange a (N, H, W) stack into a roughly square tiled 2D image."""
    n, h, w = vol.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    vmax = vol.max() if vol.max() > 0 else 1.0
    canvas = np.zeros((rows * (h + pad) - pad, cols * (w + pad) - pad))
    for i in range(n):
        r, c = divmod(i, cols)
        y, x = r * (h + pad), c * (w + pad)
        canvas[y:y + h, x:x + w] = vol[i] / vmax
    return canvas


# ---------------------------------------------------------------------------
# image visualization
# ---------------------------------------------------------------------------
def show_image(f, args):
    keys = [k for k in ("image_input", "image_grappa", "image_label") if k in f]
    n_slices = f[keys[0]].shape[0]

    all_ann = _all_annotations(f)
    # default slice: first annotated slice if any exist, else middle
    if args.slice is not None:
        sl = args.slice
    elif all_ann:
        sl = min(int(k) for k in all_ann)
    else:
        sl = n_slices // 2
    sl = int(np.clip(sl, 0, n_slices - 1))

    boxes = all_ann.get(str(sl), [])

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5.6))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        img = np.asarray(f[key][sl])
        ax.imshow(img, cmap="gray")
        ax.set_title(key)
        ax.axis("off")
        # boxes go on the label image by default; on every panel with --boxes-all
        if boxes and (key == "image_label" or args.boxes_all):
            for i, b in enumerate(boxes):
                _draw_box(ax, b, i)
            _draw_legend(ax, boxes)

    ann_slices = ",".join(sorted(all_ann, key=int)) if all_ann else "none"
    fig.suptitle(f"{os.path.basename(args.path)}  slice {sl}/{n_slices - 1}  "
                 f"shape={tuple(f[keys[0]].shape)}  annotated slices: {ann_slices}")
    fig.tight_layout()
    return fig


def _all_annotations(f):
    """Return {slice_str: [box, ...]} parsed from the file-level attribute."""
    raw = f.attrs.get("annotations")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _draw_box(ax, box, idx):
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (box["x"], box["y"]), box["width"], box["height"],
        linewidth=1.5, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    # small index tag at the box corner; full label goes in the legend
    ax.text(box["x"], box["y"] - 2, str(idx + 1),
            color="yellow", fontsize=9, fontweight="bold", va="bottom", ha="left")


def _draw_legend(ax, boxes):
    """Numbered label list under the panel so long text never overlaps anatomy."""
    lines = [f"{i + 1}. {b.get('label', '')}" for i, b in enumerate(boxes)]
    ax.text(0.0, -0.02, "\n".join(lines), transform=ax.transAxes,
            color="red", fontsize=8, va="top", ha="left")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Visualize FastMRI k-space / image HDF5 files")
    p.add_argument("path", help="path to .h5 file")
    p.add_argument("--slice", type=int, default=None, help="slice index (default: middle)")
    p.add_argument("--coil", type=int, default=None, help="kspace: show single coil")
    p.add_argument("--boxes-all", action="store_true",
                   help="image: draw annotation boxes on every panel (default: label only)")
    p.add_argument("--save", default=None, help="save figure to this path instead of showing")
    args = p.parse_args()

    with h5py.File(args.path, "r") as f:
        ftype = detect_type(f)
        print(f"[info] {args.path} -> {ftype}")
        for k in f:
            print(f"       {k}: {f[k].shape} {f[k].dtype}")
        fig = show_kspace(f, args) if ftype == "kspace" else show_image(f, args)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[saved] {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
