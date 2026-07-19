#!/usr/bin/env python3
"""Aggregate annotation boxes by label into per-label contact sheets.

Reads the manifest produced by batch_annotate.py, groups every box by its label,
and for each label renders a montage of the box crops (each with the box drawn on
a padded context window). Lets you scan how a given annotation type looks across
the whole dataset.

Usage:
  python aggregate_by_label.py                       # all labels -> data/annotated/by_label/
  python aggregate_by_label.py --manifest data/annotated/manifest.csv
  python aggregate_by_label.py --per-label 48        # max crops per sheet
  python aggregate_by_label.py --label "Meniscus Tear"   # one label only (repeatable)
  python aggregate_by_label.py --pad 0.6             # context margin (frac of box size)
  python aggregate_by_label.py --source image_label  # image to crop from

Output:
  <out>/<safe_label>.png     one contact sheet per label
  <out>/index.csv            label, box_count, shown, sheet_path
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py


SPLIT_TO_IMGDIR = {
    "train": "train/image",
    "val": "val/image",
    "leaderboard_acc4": "leaderboard/acc4/image",
    "leaderboard_acc8": "leaderboard/acc8/image",
}


def safe_name(label):
    s = re.sub(r"[^\w\-]+", "_", label).strip("_")
    return s or "unnamed"


def load_manifest(path):
    """Return {label: [box_row, ...]} where box_row is the parsed dict."""
    by_label = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k in ("slice", "x", "y", "width", "height"):
                row[k] = int(row[k])
            by_label[row["label"]].append(row)
    return by_label


def crop_with_context(img, box, pad_frac):
    """Return a padded crop around the box and the box coords within the crop."""
    h, w = img.shape
    bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]
    pad = int(round(pad_frac * max(bw, bh))) + 2
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad)
    y1 = min(h, by + bh + pad)
    crop = img[y0:y1, x0:x1]
    return crop, (bx - x0, by - y0, bw, bh)


def render_sheet(label, rows, root, source, per_label, pad_frac, out_path):
    """Render up to per_label crops for one label into a grid montage."""
    shown = rows[:per_label]
    n = len(shown)
    cols = int(math.ceil(math.sqrt(n)))
    rows_n = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.8, rows_n * 1.9))
    axes = np.atleast_1d(axes).ravel()

    # cache open files across crops of this label to avoid reopening per box
    cache = {}
    try:
        for i, r in enumerate(shown):
            ax = axes[i]
            key = (r["split"], r["file"])
            if key not in cache:
                rel = SPLIT_TO_IMGDIR.get(r["split"])
                fpath = os.path.join(root, rel, r["file"] + ".h5")
                cache[key] = h5py.File(fpath, "r")
            f = cache[key]
            img = np.asarray(f[source][r["slice"]])
            crop, (cx, cy, cw, ch) = crop_with_context(img, r, pad_frac)
            ax.imshow(crop, cmap="gray")
            ax.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=1.2,
                                           edgecolor="red", facecolor="none"))
            ax.set_title(f"{r['file']} s{r['slice']}", fontsize=5)
            ax.axis("off")
    finally:
        for f in cache.values():
            f.close()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    trunc = "" if n == len(rows) else f"  (showing {n}/{len(rows)})"
    fig.suptitle(f"{label}   —   {len(rows)} boxes{trunc}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return n


def main():
    p = argparse.ArgumentParser(description="Aggregate annotation boxes by label")
    p.add_argument("--manifest", default="data/annotated/manifest.csv")
    p.add_argument("--root", default="data/Data", help="dataset root")
    p.add_argument("--out", default="data/annotated/by_label")
    p.add_argument("--source", default="image_label",
                   choices=["image_label", "image_input", "image_grappa"])
    p.add_argument("--per-label", type=int, default=64,
                   help="max crops per contact sheet (default 64)")
    p.add_argument("--pad", type=float, default=0.5,
                   help="context margin as fraction of box size (default 0.5)")
    p.add_argument("--label", action="append", help="limit to label(s), repeatable")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[error] manifest not found: {args.manifest} "
              f"(run batch_annotate.py first)", file=sys.stderr)
        sys.exit(1)

    by_label = load_manifest(args.manifest)
    wanted = set(args.label) if args.label else None
    os.makedirs(args.out, exist_ok=True)

    index = []
    labels = sorted(by_label, key=lambda k: -len(by_label[k]))
    for label in labels:
        if wanted and label not in wanted:
            continue
        rows = by_label[label]
        out_path = os.path.join(args.out, safe_name(label) + ".png")
        shown = render_sheet(label, rows, args.root, args.source,
                             args.per_label, args.pad, out_path)
        index.append([label, len(rows), shown, out_path])
        print(f"  {label:45s} {len(rows):4d} boxes -> {shown} shown")

    idx_path = os.path.join(args.out, "index.csv")
    with open(idx_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "box_count", "shown", "sheet_path"])
        w.writerows(index)

    print(f"[done] {len(index)} label sheets -> {args.out}")
    print(f"[done] index: {idx_path}")


if __name__ == "__main__":
    main()
