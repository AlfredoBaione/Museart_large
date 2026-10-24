"""
2_music_vectors.py -- the pieces of music and their ImageBind vectors (audio
encoder).

Pieces of 10 s: one every 2 minutes of music and at least 3 per track, centred
in equal parts of the track, on the durations of track_durations.csv (measured
by decoding every file). A piece is decoded from the first channel at 16 kHz
and cut into 3 clips of 2 s, as in imagebind/data.py; pieces whose mean
amplitude is below -60 dBFS (the silence rule of preprocess_example.py) get no
vector. Tracks are taken in the order of metadata_music.csv, in chunks of 100;
every finished chunk is saved in OUTPUT/vectors/music_<n>.npz, so that a new
run starts from the first missing chunk. At the end:
OUTPUT/music_vectors_imagebind.npy (float32, one row per kept piece) and
OUTPUT/music_pieces.csv (all the planned pieces, with their status and their
row in the vectors, or -1).

Usage:
    python 2_music_vectors.py --data D:/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
"""

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import common

CHUNK, BATCH, THREADS = 100, 64, 6

args = common.parse_args(__doc__.splitlines()[1], imagebind=True)
out = args.output / "vectors"
log = open(args.output / "music_vectors_imagebind.log", "a", encoding="utf-8", buffering=1)
durations = {r["file"]: float(r["seconds"]) for r in common.read_csv(common.HERE / "track_durations.csv")}
tracks = common.read_csv(args.data / "metadata_music.csv")


def piece(job):
    path, start = job
    samples, rc, err = common.decode_piece(path, start, common.PIECE_S, common.IB_SAMPLE_RATE)
    if len(samples) < 2 * common.IB_SAMPLE_RATE:
        return None, len(samples), f"short ({len(samples)} samples, FFmpeg {rc}: {err.strip()[:120]})"
    mean_dbfs = 20 * math.log10(float(np.mean(np.abs(samples))) + 1e-7)
    if mean_dbfs < common.SILENCE_DB:
        return None, len(samples), f"silent ({mean_dbfs:.1f} dBFS)"
    clips = common.piece_to_clips(torch.from_numpy(samples).unsqueeze(0))
    return clips, len(samples), f"ok, FFmpeg {rc}" if rc else "ok"


chunks = [tracks[i:i + CHUNK] for i in range(0, len(tracks), CHUNK)]
todo = [k for k in range(len(chunks)) if not (out / f"music_{k:03d}.npz").exists()]
total = sum(len(common.plan(durations[t["file"]])) for t in tracks)
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} start: {len(tracks)} tracks, {total} pieces planned, '
          f'{len(chunks)} chunks, {len(todo)} to do\n')
model = common.load_imagebind(args.imagebind, args.checkpoint) if todo else None
with ThreadPoolExecutor(THREADS) as ex:
    for k in todo:
        t0 = time.time()
        jobs = [(t, i, s) for t in chunks[k] for i, s in enumerate(common.plan(durations[t["file"]]))]
        results = ex.map(piece, [(args.data / t["file"], s) for t, _, s in jobs])
        vectors, rows, kept, batch = [], [], [], []
        for (t, i, s), (clips, n, status) in zip(jobs, results):
            rows.append([t["file"], i, f"{s:.3f}", f"{n / common.IB_SAMPLE_RATE:.3f}", status])
            if clips is not None:
                batch.append(clips)
                kept.append(len(rows) - 1)  # vectors keep the order of the kept pieces
            if len(batch) == BATCH:
                vectors.append(common.embed_imagebind(model, common.AUDIO, torch.stack(batch)))
                batch = []
        if batch:
            vectors.append(common.embed_imagebind(model, common.AUDIO, torch.stack(batch)))
        vectors = np.concatenate(vectors) if vectors else np.zeros((0, 1024), np.float32)
        np.savez(out / f"music_{k:03d}.tmp.npz", vectors=vectors, kept=np.array(kept, dtype=np.int64),
                 rows=np.array(rows, dtype=object))
        (out / f"music_{k:03d}.tmp.npz").replace(out / f"music_{k:03d}.npz")
        bad = sum(1 for r in rows if not r[4].startswith("ok"))
        log.write(f'{time.strftime("%H:%M:%S")} chunk {k + 1}/{len(chunks)}: {len(rows)} pieces, {bad} without vector, '
                  f'{time.time() - t0:.0f} s\n')

all_vectors, out_rows, offset = [], [], 0
for k in range(len(chunks)):
    z = np.load(out / f"music_{k:03d}.npz", allow_pickle=True)
    row_of = {j: offset + n for n, j in enumerate(z["kept"].tolist())}
    assert len(row_of) == len(z["vectors"])
    offset += len(row_of)
    all_vectors.append(z["vectors"])
    for j, r in enumerate(z["rows"].tolist()):
        out_rows.append(r + [row_of.get(j, -1)])
vectors = np.concatenate(all_vectors)
np.save(args.output / "music_vectors_imagebind.npy", vectors)
meta = {t["file"]: (t["class"], t["split"]) for t in tracks}
with open(args.output / "music_pieces.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "split", "piece", "start_s", "decoded_s", "status", "vector_row"])
    for r in out_rows:
        w.writerow([r[0], *meta[r[0]], *r[1:]])
statuses = {}
for r in out_rows:
    key = r[4].split(" ")[0].rstrip(",")
    statuses[key] = statuses.get(key, 0) + 1
log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} done: {len(out_rows)} pieces, {vectors.shape[0]} vectors, '
          f'status {statuses}\n')
log.close()
