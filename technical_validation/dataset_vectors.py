"""
dataset_vectors.py -- the vectors published with the dataset, in the files read by 4_analysis.py.

Reads the .npz file of every image and track of the metadata and writes OUTPUT/image_vectors.npy,
image_index.csv, music_vectors_imagebind.npy, music_vectors_clap.npy and music_pieces.csv.

Usage:
    python dataset_vectors.py --data D:/Museart_Large
"""

import csv

import numpy as np

import common

args = common.parse_args(__doc__.splitlines()[1])


def vectors(row):
    return np.load(args.data / (row["file"].rsplit(".", 1)[0] + ".npz"))


images = common.read_csv(args.data / "metadata_images.csv")
np.save(args.output / "image_vectors.npy", np.stack([vectors(r)["imagebind"] for r in images]))
with open(args.output / "image_index.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "split"])
    w.writerows([r["file"], r["class"], r["split"]] for r in images)

pieces, imagebind, clap = [], [], []
for r in common.read_csv(args.data / "metadata_music.csv"):
    z = vectors(r)
    pieces += [[r["file"], r["class"], r["split"], f"{s:.3f}", len(pieces) + i] for i, s in enumerate(z["start_s"])]
    imagebind.append(z["imagebind"])
    clap.append(z["clap"])
np.save(args.output / "music_vectors_imagebind.npy", np.concatenate(imagebind))
np.save(args.output / "music_vectors_clap.npy", np.concatenate(clap))
with open(args.output / "music_pieces.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "split", "start_s", "vector_row"])
    w.writerows(pieces)
print(f"{len(images)} images, {len(pieces)} pieces of {len(imagebind)} tracks")
