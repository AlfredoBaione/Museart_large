"""
2_music_vectors.py -- ImageBind and CLAP vectors of the pieces of 10 s of the segments, computed from the audio files.

Every track is decoded once by FFmpeg from its start (first channel, 16 kHz and 48 kHz) and read in segments of
30 s; each segment of metadata_music.csv gives its three pieces of 10 s, and a silent piece gets no vector.
ImageBind: 3 clips of 2 s per piece; CLAP: the features of its feature extractor, computed on the GPU. One file
per track in OUTPUT/vectors/, written when the track is complete (a new run skips the tracks already written);
then OUTPUT/music_vectors_imagebind.npy, music_vectors_clap.npy and music_pieces.csv.

Usage:
    python 2_music_vectors.py --data D:/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
"""

import csv
import queue
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import common

BATCH, THREADS = 64, 6
SR16, SR48 = common.IB_SAMPLE_RATE, common.CLAP_SAMPLE_RATE
PIECES = common.SEGMENT_S // common.PIECE_S

args = common.parse_args(__doc__.splitlines()[1], imagebind=True)
log = open(args.output / "music_vectors.log", "a", encoding="utf-8", buffering=1)

tracks, segments = {}, defaultdict(set)
for r in common.read_csv(args.data / "metadata_music.csv"):
    track, n = r["file"].rsplit("_", 1)
    tracks.setdefault(track, r)
    segments[track].add(int(n))


def audio_file(track):
    found = [p for p in (args.data / track).parent.glob((args.data / track).name + ".*") if p.suffix != ".npz"]
    assert len(found) == 1, (track, found)
    return found[0]


def target(track):
    return args.output / "vectors" / (track + ".npz")


def say(text):
    log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {text}\n')


todo = sorted((t for t in tracks if not target(t).exists()), key=lambda t: -audio_file(t).stat().st_size)
say(f"start: {len(tracks)} tracks, {len(todo)} to do")
q = queue.Queue(maxsize=2 * BATCH)


def worker(i):
    track = todo[i]
    procs = []
    try:
        (p16, e16), (p48, e48) = (common.decode_stream(audio_file(track), sr) for sr in (SR16, SR48))
        procs = [p16, p48]
        n, sent = 0, 0
        while True:
            b16 = common.read_samples(p16.stdout, common.SEGMENT_S * SR16)
            b48 = common.read_samples(p48.stdout, common.SEGMENT_S * SR48)
            if len(b16) < common.SEGMENT_S * SR16 or len(b48) < common.SEGMENT_S * SR48:
                break
            n += 1
            if n not in segments[track]:
                continue
            for k in range(PIECES):
                s16 = b16[k * common.PIECE_S * SR16:(k + 1) * common.PIECE_S * SR16]
                if common.is_silent(s16):
                    continue
                clips = common.piece_to_clips(torch.from_numpy(s16.copy()).unsqueeze(0))
                q.put(("piece", i, (n, k + 1), clips, b48[k * common.PIECE_S * SR48:(k + 1) * common.PIECE_S * SR48]))
                sent += 1
        messages = [common.end_stream(p, e) for p, e in ((p16, e16), (p48, e48))]
        q.put(("done", i, sent, messages))
    except Exception as e:  # noqa: BLE001
        for p in procs:
            p.kill()
        q.put(("failed", i, repr(e)))


ib_model = common.load_imagebind(args.imagebind, args.checkpoint, modalities=(common.AUDIO,)) if todo else None
clap_model = common.load_clap() if todo else None
clap_features = common.ClapFeaturesGPU() if todo else None
results, expected, failed = defaultdict(list), {}, set()


def embed(batch):
    ib = common.embed_imagebind(ib_model, common.AUDIO, torch.stack([b[3] for b in batch]))
    cl = common.embed_clap(clap_model, *clap_features(torch.from_numpy(np.stack([b[4] for b in batch])).cuda()))
    for b, v_ib, v_cl in zip(batch, ib, cl):
        if b[1] not in failed:
            results[b[1]].append((b[2], v_ib, v_cl))


def save(i):
    rows = sorted(results.pop(i, []), key=lambda r: r[0])
    path = target(todo[i])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, imagebind=np.array([r[1] for r in rows], np.float32).reshape(-1, 1024),
             clap=np.array([r[2] for r in rows], np.float32).reshape(-1, 512),
             segment=np.array([r[0][0] for r in rows], np.int64), piece=np.array([r[0][1] for r in rows], np.int64))
    tmp.replace(path)


def complete():
    for i in [i for i in expected if len(results[i]) == expected[i]]:
        del expected[i]
        save(i)


t0, last = time.time(), time.time()
with ThreadPoolExecutor(THREADS) as ex:
    futures = [ex.submit(worker, i) for i in range(len(todo))]
    batch, written = [], 0
    while True:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            if all(f.done() for f in futures) and q.empty():
                break
            continue
        if item[0] == "piece":
            batch.append(item)
            if len(batch) == BATCH:
                embed(batch)
                batch = []
        elif item[0] == "done":
            expected[item[1]] = item[2]
            for rc, text in item[3]:
                if rc or text:
                    say(f"{todo[item[1]]}: FFmpeg {rc} {' '.join(text.split())[:200]}")
        else:
            failed.add(item[1])
            results.pop(item[1], None)
            say(f"FAILED {todo[item[1]]}: {item[2]}")
        if not batch or len(batch) == BATCH:
            before = len(expected)
            complete()
            written += before - len(expected)
        if time.time() - last > 60:
            last = time.time()
            say(f"{written}/{len(todo)} tracks, {(time.time() - t0) / 60:.0f} min")
    if batch:
        embed(batch)
    complete()
say(f"end: {len(todo) - len(expected) - len(failed)} tracks written in {(time.time() - t0) / 60:.1f} min, "
    f"failed {len(failed)}")

missing = [t for t in tracks if not target(t).exists()]
assert not missing, f"{len(missing)} tracks without vectors (see the log), e.g. {missing[:3]}; run the script again"
rows = []
for track, r in tracks.items():
    with np.load(target(track)) as z:
        rows += [[track, f"{track}_{n:03d}", k, r["class"], r["split"]] for n, k in zip(z["segment"], z["piece"])]
ib = np.lib.format.open_memmap(args.output / "music_vectors_imagebind.npy", "w+", np.float32, (len(rows), 1024))
cl = np.lib.format.open_memmap(args.output / "music_vectors_clap.npy", "w+", np.float32, (len(rows), 512))
i = 0
for track in tracks:
    with np.load(target(track)) as z:
        ib[i:i + len(z["piece"])], cl[i:i + len(z["piece"])] = z["imagebind"], z["clap"]
        i += len(z["piece"])
ib.flush()
cl.flush()
with open(args.output / "music_pieces.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track", "segment", "piece", "class", "split", "vector_row"])
    w.writerows(r + [j] for j, r in enumerate(rows))
say(f"done: {len(rows)} pieces of {len(tracks)} tracks")
log.close()
