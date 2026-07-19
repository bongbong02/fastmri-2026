#!/usr/bin/env python3
"""Batch-render annotated label images for the whole FastMRI dataset.

Walks every image/*.h5 file under the dataset root, and for each annotated slice
saves a PNG of image_label with the bounding boxes and a numbered legend drawn on
it. Output mirrors the split layout, plus a CSV manifest of every box.

Layout handled (any that exist):
  <root>/train/image/*.h5
  <root>/val/image/*.h5
  <root>/leaderboard/acc4/image/*.h5
  <root>/leaderboard/acc8/image/*.h5

Usage:
  python batch_annotate.py                          # all splits -> data/annotated/
  python batch_annotate.py --root data/Data --out data/annotated
  python batch_annotate.py --split train            # one split only
  python batch_annotate.py --all-slices             # every slice, not just annotated
  python batch_annotate.py --source image_input     # box on input instead of label
  python batch_annotate.py --workers 8              # parallel rendering

Output:
  <out>/<split>/<file_stem>_sl<NN>.png
  <out>/manifest.csv    (split, file, slice, box_index, label, x, y, w, h)
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for batch rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py


# splits to scan: (label, glob-relative image dir)
SPLITS = {
    "train": "train/image",
    "val": "val/image",
    "leaderboard_acc4": "leaderboard/acc4/image",
    "leaderboard_acc8": "leaderboard/acc8/image",
}


def parse_annotations(f):
    raw = f.attrs.get("annotations")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def render_slice(img, boxes, title, out_path):
    """Render one label slice with boxes + numbered legend, save to out_path."""
    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for i, b in enumerate(boxes):
        rect = patches.Rectangle(
            (b["x"], b["y"]), b["width"], b["height"],
            linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(b["x"], b["y"] - 2, str(i + 1),
                color="yellow", fontsize=9, fontweight="bold", va="bottom", ha="left")
    if boxes:
        lines = [f"{i + 1}. {b.get('label', '')}" for i, b in enumerate(boxes)]
        ax.text(0.0, -0.02, "\n".join(lines), transform=ax.transAxes,
                color="red", fontsize=8, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_file(fpath, split, out_dir, source, all_slices):
    """Render every requested slice of one file. Returns list of manifest rows."""
    rows = []
    stem = os.path.splitext(os.path.basename(fpath))[0]
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    with h5py.File(fpath, "r") as f:
        if source not in f:
            return rows
        vol = f[source]
        n_slices = vol.shape[0]
        ann = parse_annotations(f)

        slices = range(n_slices) if all_slices else sorted(
            (int(k) for k in ann), key=int)
        for sl in slices:
            if sl < 0 or sl >= n_slices:
                continue
            boxes = ann.get(str(sl), [])
            if not all_slices and not boxes:
                continue
            img = np.asarray(vol[sl])
            title = f"{stem}  {source}  slice {sl}/{n_slices - 1}  ({len(boxes)} box)"
            out_path = os.path.join(split_out, f"{stem}_sl{sl:02d}.png")
            render_slice(img, boxes, title, out_path)
            for i, b in enumerate(boxes):
                rows.append([split, stem, sl, i + 1, b.get("label", ""),
                             b["x"], b["y"], b["width"], b["height"]])
    return rows


def _worker(job):
    fpath, split, out_dir, source, all_slices = job
    try:
        return process_file(fpath, split, out_dir, source, all_slices)
    except Exception as e:  # keep the batch alive on a single bad file
        print(f"[warn] failed {fpath}: {e}", file=sys.stderr)
        return []


def collect_jobs(root, which_splits, out_dir, source, all_slices):
    import glob
    jobs = []
    for split, rel in SPLITS.items():
        if which_splits and split not in which_splits:
            continue
        img_dir = os.path.join(root, rel)
        if not os.path.isdir(img_dir):
            continue
        for fp in sorted(glob.glob(os.path.join(img_dir, "*.h5"))):
            jobs.append((fp, split, out_dir, source, all_slices))
    return jobs


def main():
    p = argparse.ArgumentParser(description="Batch-render annotated label images")
    p.add_argument("--root", default="data/Data", help="dataset root (default: data/Data)")
    p.add_argument("--out", default="data/annotated", help="output dir (default: data/annotated)")
    p.add_argument("--split", action="append",
                   help=f"limit to split(s): {', '.join(SPLITS)} (repeatable)")
    p.add_argument("--source", default="image_label",
                   choices=["image_label", "image_input", "image_grappa"],
                   help="which image the boxes are drawn on (default: image_label)")
    p.add_argument("--all-slices", action="store_true",
                   help="render every slice, not just annotated ones")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel worker processes")
    args = p.parse_args()

    jobs = collect_jobs(args.root, set(args.split or []), args.out, args.source, args.all_slices)
    if not jobs:
        print(f"[error] no image .h5 files found under {args.root}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"[info] {len(jobs)} files, source={args.source}, "
          f"{'all slices' if args.all_slices else 'annotated slices only'}, "
          f"workers={args.workers}")

    all_rows = []
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for fut in as_completed(futs):
                all_rows.extend(fut.result())
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} files, {len(all_rows)} boxes")
    else:
        for j in jobs:
            all_rows.extend(_worker(j))
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files, {len(all_rows)} boxes")

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "file", "slice", "box_index", "label", "x", "y", "width", "height"])
        w.writerows(sorted(all_rows, key=lambda r: (r[0], r[1], r[2], r[3])))

    n_imgs = len({(r[0], r[1], r[2]) for r in all_rows})
    print(f"[done] {len(all_rows)} boxes across {n_imgs} slices -> {args.out}")
    print(f"[done] manifest: {manifest}")


if __name__ == "__main__":
    main()
