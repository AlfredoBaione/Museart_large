"""
1_image_vectors.py -- ImageBind vectors of the images, computed from the image files.

Writes OUTPUT/image_vectors.npy (one row per row of metadata_images.csv) and OUTPUT/image_index.csv; an image that
cannot be read gets a row of NaN. A new run starts from the first missing chunk of 1024 images.

Usage:
    python 1_image_vectors.py --data D:/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import common

CHUNK, BATCH, THREADS = 1024, 32, 4

args = common.parse_args(__doc__.splitlines()[1], imagebind=True)
out = args.output / "vectors"
log = open(args.output / "image_vectors.log", "a", encoding="utf-8", buffering=1)
rows = common.read_csv(args.data / "metadata_images.csv")
chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
todo = [k for k in range(len(chunks)) if not (out / f"images_{k:03d}.npy").exists()]
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start: {len(rows)} images, {len(chunks)} chunks, {len(todo)} to do\n')


def load(row):
    try:
        return common.load_image(args.data / row["file"]), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


model = common.load_imagebind(args.imagebind, args.checkpoint) if todo else None
with ThreadPoolExecutor(THREADS) as ex:
    pending = ex.map(load, chunks[todo[0]]) if todo else None
    for j, k in enumerate(todo):
        t = time.time()
        loaded = list(pending)
        if j + 1 < len(todo):
            pending = ex.map(load, chunks[todo[j + 1]])  # read the next chunk during the GPU work
        vectors = np.full((len(loaded), 1024), np.nan, dtype=np.float32)
        good = [i for i, (x, _) in enumerate(loaded) if x is not None]
        for b in range(0, len(good), BATCH):
            idx = good[b:b + BATCH]
            vectors[idx] = common.embed_imagebind(model, common.VISION, torch.stack([loaded[i][0] for i in idx]))
        errors = [(chunks[k][i]["file"], err) for i, (x, err) in enumerate(loaded) if x is None]
        if errors:
            with open(out / f"images_{k:03d}_errors.csv", "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(errors)
        np.save(out / f"images_{k:03d}.tmp.npy", vectors)
        (out / f"images_{k:03d}.tmp.npy").replace(out / f"images_{k:03d}.npy")
        log.write(f'{time.strftime("%H:%M:%S")} chunk {k + 1}/{len(chunks)}: {len(loaded)} images, '
                  f'{len(errors)} unreadable, {time.time() - t:.0f} s\n')

vectors = np.concatenate([np.load(out / f"images_{k:03d}.npy") for k in range(len(chunks))])
np.save(args.output / "image_vectors.npy", vectors)
with open(args.output / "image_index.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "split"])
    w.writerows([r["file"], r["class"], r["split"]] for r in rows)
errors = []
for k in range(len(chunks)):
    p = out / f"images_{k:03d}_errors.csv"
    if p.exists():
        errors += list(csv.reader(open(p, encoding="utf-8")))
with open(args.output / "image_errors.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "error"])
    w.writerows(errors)
norms = np.linalg.norm(vectors, axis=1)
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} done: {vectors.shape[0]} rows, {int(np.isnan(norms).sum())} NaN rows, '
          f'{len(errors)} unreadable\n')
log.close()
