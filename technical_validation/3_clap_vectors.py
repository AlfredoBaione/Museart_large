"""
3_clap_vectors.py -- CLAP vectors of the pieces that have an ImageBind vector, computed from the audio files.

Row i of OUTPUT/music_vectors_clap.npy is the piece of row i of OUTPUT/music_vectors_imagebind.npy; a piece that
cannot be decoded gets a row of NaN. Run 2_music_vectors.py first.

Usage:
    python 3_clap_vectors.py --data D:/Museart_Large
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common

CHUNK, BATCH, THREADS = 2000, 64, 6

args = common.parse_args(__doc__.splitlines()[1])
out = args.output / "vectors"
log = open(args.output / "music_vectors_clap.log", "a", encoding="utf-8", buffering=1)
pieces = sorted((r for r in common.read_csv(args.output / "music_pieces.csv") if int(r["vector_row"]) >= 0),
                key=lambda r: int(r["vector_row"]))
assert [int(r["vector_row"]) for r in pieces] == list(range(len(pieces)))
chunks = [pieces[i:i + CHUNK] for i in range(0, len(pieces), CHUNK)]
todo = [k for k in range(len(chunks)) if not (out / f"clap_{k:03d}.npy").exists()]
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start: {common.CLAP_CHECKPOINT} {common.CLAP_REVISION[:7]}, '
          f'{len(pieces)} pieces, {len(chunks)} chunks, {len(todo)} to do\n')


def piece(row):
    samples, rc, err = common.decode_piece(args.data / row["file"], float(row["start_s"]), common.PIECE_S,
                                           common.CLAP_SAMPLE_RATE)
    if len(samples) < common.CLAP_SAMPLE_RATE:
        return None, f"short ({len(samples)} samples, FFmpeg {rc}: {err.strip()[:120]})"
    return common.clap_features(samples), ""


model = common.load_clap() if todo else None
with ThreadPoolExecutor(THREADS) as ex:
    for k in todo:
        t0 = time.time()
        vectors = np.full((len(chunks[k]), 512), np.nan, dtype=np.float32)
        batch, rows, errors = [], [], []
        for i, (feats, err) in enumerate(ex.map(piece, chunks[k])):
            if feats is None:
                errors.append((chunks[k][i]["file"], chunks[k][i]["piece"], err))
            else:
                batch.append(feats)
                rows.append(i)
            if len(batch) == BATCH or (i == len(chunks[k]) - 1 and batch):
                vectors[rows] = common.embed_clap(model, np.concatenate([f[0] for f in batch]),
                                                  np.concatenate([f[1] for f in batch]))
                batch, rows = [], []
        if errors:
            with open(out / f"clap_{k:03d}_errors.csv", "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(errors)
        np.save(out / f"clap_{k:03d}.tmp.npy", vectors)
        (out / f"clap_{k:03d}.tmp.npy").replace(out / f"clap_{k:03d}.npy")
        log.write(f'{time.strftime("%H:%M:%S")} chunk {k + 1}/{len(chunks)}: {len(chunks[k])} pieces, '
                  f'{len(errors)} not decoded, {time.time() - t0:.0f} s\n')

vectors = np.concatenate([np.load(out / f"clap_{k:03d}.npy") for k in range(len(chunks))])
np.save(args.output / "music_vectors_clap.npy", vectors)
errors = []
for k in range(len(chunks)):
    p = out / f"clap_{k:03d}_errors.csv"
    if p.exists():
        errors += list(csv.reader(open(p, encoding="utf-8")))
with open(args.output / "clap_errors.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "piece", "error"])
    w.writerows(errors)
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} done: {vectors.shape[0]} rows, '
          f'{int(np.isnan(vectors).any(axis=1).sum())} NaN rows\n')
log.close()
