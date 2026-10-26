"""
4_analysis.py -- the cross-modal test and the classifiers, on the vectors in OUTPUT; a few minutes on the CPU.

Cross-modal test (ImageBind, nothing trained): rank of the right class among the 41 class centroids and share of
the same class among the 10 nearest files, with permutation tests. Classifiers: logistic regression on standardized
vectors, C chosen on the macro F1 of val, results on test; a track is predicted by the mean probability of its
pieces. Writes results.txt, classes.csv, similarity.csv and confusion_*.csv in OUTPUT.

Usage:
    python 4_analysis.py
"""

import copy
import csv
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

import common

PERMUTATIONS, K, SEED = 10000, 10, 0
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PERIODS = ["Pre-renaissance", "Flemish", "Renaissance", "Baroque", "Classical", "Romanticism", "Late romanticism",
           "Early 20th century", "20th century"]
HISTORICAL = ["Pre-renaissance sacred", "Pre-renaissance secular", "Flemish sacred", "Flemish secular",
              "Renaissance sacred", "Renaissance secular", "Baroque sacred", "Baroque secular", "Classical sacred",
              "Classical secular", "Romanticism chamber", "Romanticism orchestral", "Late romanticism chamber",
              "Late romanticism orchestral", "Early 20th century traditional chamber",
              "Early 20th century traditional orchestral", "Early 20th century experimental chamber",
              "Early 20th century experimental orchestral", "20th century traditional chamber",
              "20th century traditional orchestral", "20th century experimental chamber",
              "20th century experimental orchestral"]
CULTURAL = ["African traditional folklore", "Afro-American", "East Asian traditional folklore", "East European folklore",
            "Latin-American", "Middle, north European folklore", "Middle, south Asian traditional folklore",
            "North American traditional folklore", "South American traditional folklore", "South European folklore"]
STYLISTIC = ["Country", "Dance", "Electronic", "Heavy metal", "Jazz", "Pop", "Rap, hip hop", "Rock",
             "Western songwriting"]
ORDER = HISTORICAL + CULTURAL + STYLISTIC
GROUP = np.array([0] * len(HISTORICAL) + [1] * len(CULTURAL) + [2] * len(STYLISTIC))
PERIOD = np.array([max((p for p in PERIODS if c.startswith(p)), key=len) for c in HISTORICAL] + [""] * 19)
N = len(ORDER)

args = common.parse_args(__doc__.splitlines()[1])
log = open(args.output / "analysis.log", "a", encoding="utf-8", buffering=1)
out = open(args.output / "results.txt", "w", encoding="utf-8", buffering=1)
rng = np.random.default_rng(SEED)


def say(line=""):
    out.write(line + "\n")
    log.write(f'{time.strftime("%H:%M:%S")} {line}\n')


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


img = np.load(args.output / "image_vectors.npy")
img_rows = common.read_csv(args.output / "image_index.csv")
assert len(img_rows) == len(img) and not np.isnan(img).any()
img_y = np.array([ORDER.index(r["class"]) for r in img_rows])
img_split = np.array([r["split"] for r in img_rows])

pieces = [r for r in common.read_csv(args.output / "music_pieces.csv") if int(r["vector_row"]) >= 0]
mus = np.load(args.output / "music_vectors_imagebind.npy")
clap = np.load(args.output / "music_vectors_clap.npy")
assert len(pieces) == len(mus) and [int(r["vector_row"]) for r in pieces] == list(range(len(mus)))
assert clap.shape == (len(mus), 512) and not np.isnan(clap).any()
piece_y = np.array([ORDER.index(r["class"]) for r in pieces])
piece_split = np.array([r["split"] for r in pieces])
by_track = defaultdict(list)
for i, r in enumerate(pieces):
    by_track[r["file"]].append(i)
track_files = list(by_track)
track_rows = {r["file"]: r for r in pieces}
track_y = np.array([ORDER.index(track_rows[f]["class"]) for f in track_files])
track_split = np.array([track_rows[f]["split"] for f in track_files])
track_vec = normalize(np.stack([mus[by_track[f]].mean(axis=0) for f in track_files]))
say(f"images {len(img)}; music: {len(mus)} pieces with a vector, {len(track_files)} tracks")

c_img = normalize(np.stack([img[img_y == k].mean(axis=0) for k in range(N)]))
c_mus = normalize(np.stack([track_vec[track_y == k].mean(axis=0) for k in range(N)]))
S = c_mus @ c_img.T


def class_stats(s):
    """Statistics of a music x image similarity table whose diagonal holds the right pairs."""
    diag = np.diag(s)
    rank_m2i = 1 + (s > diag[:, None]).sum(axis=1)
    rank_i2m = 1 + (s > diag[None, :]).sum(axis=0)
    nearest = s.argmax(axis=1)
    hist = GROUP == 0
    return {"mean rank music->image": rank_m2i.mean(), "mean rank image->music": rank_i2m.mean(),
            "right class first, music->image": (rank_m2i == 1).sum(),
            "right class in top 5, music->image": (rank_m2i <= 5).sum(),
            "right class first, image->music": (rank_i2m == 1).sum(),
            "right class in top 5, image->music": (rank_i2m <= 5).sum(),
            "nearest image class in the same group": (GROUP[nearest] == GROUP).sum(),
            "nearest image class in the same period (historical music classes)":
                (PERIOD[nearest][hist] == PERIOD[hist]).sum()}, rank_m2i, rank_i2m, nearest


observed, rank_m2i, rank_i2m, nearest = class_stats(S)
null = defaultdict(list)
for _ in range(PERMUTATIONS):
    p = rng.permutation(N)
    for key, v in class_stats(S[:, p])[0].items():  # image class b gets the centroid of class p[b]
        null[key].append(v)
say("CROSS-MODAL, CLASS LEVEL (ImageBind, 41 classes, all files; chance: mean rank 21)")
for key, v in observed.items():
    nv = np.array(null[key])
    lower_is_better = key.startswith("mean rank")
    pval = (1 + (nv <= v).sum() if lower_is_better else 1 + (nv >= v).sum()) / (PERMUTATIONS + 1)
    say(f"  {key}: {v:.2f} (permutations: mean {nv.mean():.2f}, 95% range {np.percentile(nv, 2.5):.1f}-"
        f"{np.percentile(nv, 97.5):.1f}; p = {pval:.4f})")
say(f"  mean cosine, right pairs {np.diag(S).mean():.4f}; other pairs {S[~np.eye(N, dtype=bool)].mean():.4f}")
with open(args.output / "similarity.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["music class \\ image class"] + ORDER)
    w.writerows([ORDER[a]] + [f"{x:.5f}" for x in S[a]] for a in range(N))



def top_k(queries, gallery, chunk=256):
    """Indices of the K most similar gallery vectors, in any order."""
    idx = np.empty((len(queries), K), dtype=np.int64)
    for i in range(0, len(queries), chunk):
        idx[i:i + chunk] = np.argpartition(-(queries[i:i + chunk] @ gallery.T), K, axis=1)[:, :K]
    return idx


def file_level(name, query_y, gallery_y, top):
    queries_per_class = np.bincount(query_y, minlength=N)
    hits = (gallery_y[top] == query_y[:, None]).mean(axis=1)
    per_class = np.bincount(query_y, weights=hits, minlength=N) / queries_per_class
    chance = np.bincount(gallery_y, minlength=N) / len(gallery_y)
    nv = []
    for _ in range(1000):
        h = (rng.permutation(gallery_y)[top] == query_y[:, None]).mean(axis=1)
        nv.append((np.bincount(query_y, weights=h, minlength=N) / queries_per_class).mean())
    nv = np.array(nv)
    pval = (1 + (nv >= per_class.mean()).sum()) / (len(nv) + 1)
    say(f"  {name}: same class among the {K} nearest, mean of the classes {100 * per_class.mean():.2f}% "
        f"(chance {100 * chance.mean():.2f}%; 1000 permutations of the labels of the searched files: max "
        f"{100 * nv.max():.2f}%, p = {pval:.4f}); all queries {100 * hits.mean():.2f}% "
        f"(chance {100 * chance[query_y].mean():.2f}%); classes above chance {(per_class > chance).sum()}/{N}")
    return per_class, chance


say()
say("CROSS-MODAL, FILE LEVEL (ImageBind, all files)")
p_t2i, ch_t2i = file_level("track -> images", track_y, img_y, top_k(track_vec, img))
p_i2t, ch_i2t = file_level("image -> tracks", img_y, track_y, top_k(img, track_vec))



def fit(x_tr, y_tr, x_val, y_val):
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_val = scaler.transform(x_tr), scaler.transform(x_val)
    best = None
    clf = LogisticRegression(max_iter=3000, warm_start=True)  # C grows along the grid, from the previous solution
    for c in C_GRID:
        clf.set_params(C=c)
        clf.fit(x_tr, y_tr)
        f1 = f1_score(y_val, clf.predict(x_val), average="macro", labels=range(N), zero_division=0)
        log.write(f'{time.strftime("%H:%M:%S")}   C={c}: val macro F1 {f1:.4f}, {clf.n_iter_[0]} iterations\n')
        if best is None or f1 > best[0]:
            best = (f1, c, copy.deepcopy(clf))
    assert list(best[2].classes_) == list(range(N))
    return scaler, best[2], best[1]


def report(name, y_true, y_pred, majority):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=range(N), zero_division=0)
    say(f"  {name}: accuracy {100 * acc:.2f}% (majority class {100 * (y_true == majority).mean():.2f}%), "
        f"balanced accuracy {100 * bal:.2f}% (chance {100 / N:.2f}%), macro F1 {100 * f1:.2f}%")
    return f1_score(y_true, y_pred, average=None, labels=range(N), zero_division=0)


def save_confusion(name, y_true, y_pred):
    with open(args.output / f"confusion_{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + ORDER)
        cm = confusion_matrix(y_true, y_pred, labels=range(N))
        w.writerows([ORDER[a]] + cm[a].tolist() for a in range(N))
    wrong = y_true != y_pred
    say(f"  {name}: errors in the same group {(GROUP[y_true[wrong]] == GROUP[y_pred[wrong]]).mean() * 100:.1f}% "
        f"of {wrong.sum()} errors")


def music_classifier(name, vectors):
    """Classifier on the piece vectors; a track is predicted by the mean probability of its pieces."""
    tr, va, te = (piece_split == s for s in ("train", "val", "test"))
    scaler, clf, c = fit(vectors[tr], piece_y[tr], vectors[va], piece_y[va])
    proba = clf.predict_proba(scaler.transform(vectors))
    say(f"  {name}: C = {c}, trained on {tr.sum()} pieces of {(track_split == 'train').sum()} tracks, tested on "
        f"{te.sum()} pieces of {(track_split == 'test').sum()} tracks")
    report(f"{name} pieces, test", piece_y[te], proba[te].argmax(axis=1), Counter(piece_y[tr]).most_common(1)[0][0])
    test_tracks = [i for i in range(len(track_files)) if track_split[i] == "test"]
    pred_trk = np.array([proba[by_track[track_files[i]]].mean(axis=0).argmax() for i in test_tracks])
    f1 = report(f"{name} tracks, test", track_y[test_tracks], pred_trk,
                Counter(track_y[track_split == "train"]).most_common(1)[0][0])
    save_confusion(name.replace(" ", "_"), track_y[test_tracks], pred_trk)
    return f1


say()
say(f"CLASSIFIERS (logistic regression, C chosen on val among {C_GRID})")
tr, va, te = (img_split == s for s in ("train", "val", "test"))
scaler, clf, c = fit(img[tr], img_y[tr], img[va], img_y[va])
pred_img = clf.predict(scaler.transform(img[te]))
say(f"  images (OpenCLIP ViT-H/14): C = {c}, trained on {tr.sum()}, tested on {te.sum()}")
f1_img = report("images, test", img_y[te], pred_img, Counter(img_y[tr]).most_common(1)[0][0])
save_confusion("images", img_y[te], pred_img)
say(f"  (music: CLAP {common.CLAP_CHECKPOINT}, then ImageBind audio for comparison, on the same pieces)")
f1_clap = music_classifier("music clap", clap)
f1_ib = music_classifier("music imagebind", mus)

with open(args.output / "classes.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "group", "images", "tracks", "rank_music_to_image", "rank_image_to_music",
                "nearest_image_class", "track_to_images_same_class", "chance_images", "image_to_tracks_same_class",
                "chance_tracks", "f1_images", "f1_music_clap", "f1_music_imagebind"])
    for k in range(N):
        w.writerow([ORDER[k], ["historical", "cultural", "stylistic"][GROUP[k]], (img_y == k).sum(),
                    (track_y == k).sum(), rank_m2i[k], rank_i2m[k], ORDER[nearest[k]], f"{p_t2i[k]:.4f}",
                    f"{ch_t2i[k]:.4f}", f"{p_i2t[k]:.4f}", f"{ch_i2t[k]:.4f}", f"{f1_img[k]:.4f}",
                    f"{f1_clap[k]:.4f}", f"{f1_ib[k]:.4f}"])
say()
say("done")
out.close()
log.close()
