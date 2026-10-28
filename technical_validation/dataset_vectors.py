"""
dataset_vectors.py -- the vectors published with the dataset, in the files read by 3_analysis.py.

Reads the ib_ file of every image of metadata_images.csv and the ib_ and c_ files of the pieces of every segment of
metadata_music.csv, and writes OUTPUT/image_vectors.npy, image_index.csv, music_vectors_imagebind.npy,
music_vectors_clap.npy and music_pieces.csv.

Usage:
    python dataset_vectors.py --data D:/Museart_Large
"""

import csv

import numpy as np

import common

args = common.parse_args(__doc__.splitlines()[1])


def vector_file(path, prefix, piece=None):
    folder, name = path.rsplit("/", 1)
    return args.data / folder / f"{prefix}_{name.split('.')[0]}{'' if piece is None else f'_{piece}'}.npz"


images = common.read_csv(args.data / "metadata_images.csv")
vectors = np.lib.format.open_memmap(args.output / "image_vectors.npy", "w+", np.float32, (len(images), 1024))
for i, r in enumerate(images):
    vectors[i] = np.load(vector_file(r["file"], "ib"))["imagebind"]
vectors.flush()
with open(args.output / "image_index.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "split"])
    w.writerows([r["file"], r["class"], r["split"]] for r in images)

pieces = []
for r in common.read_csv(args.data / "metadata_music.csv"):
    for k in range(1, common.SEGMENT_S // common.PIECE_S + 1):
        if vector_file(r["file"], "ib", k).exists():
            pieces.append([r["file"].rsplit("_", 1)[0], r["file"], k, r["class"], r["split"], len(pieces)])
ib = np.lib.format.open_memmap(args.output / "music_vectors_imagebind.npy", "w+", np.float32, (len(pieces), 1024))
cl = np.lib.format.open_memmap(args.output / "music_vectors_clap.npy", "w+", np.float32, (len(pieces), 512))
for i, (_, segment, k, *_) in enumerate(pieces):
    ib[i] = np.load(vector_file(segment, "ib", k))["imagebind"]
    cl[i] = np.load(vector_file(segment, "c", k))["clap"]
ib.flush()
cl.flush()
with open(args.output / "music_pieces.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track", "segment", "piece", "class", "split", "vector_row"])
    w.writerows(pieces)
print(f"{len(images)} images, {len(pieces)} pieces of {len({p[1] for p in pieces})} segments")
