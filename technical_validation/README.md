# Technical validation

The code and the results of the technical validation of Museart_Large. Three
experiments, on the release of the dataset on Kaggle:

| Experiment | Model | Question |
|---|---|---|
| Cross-modal test | ImageBind huge (Girdhar et al., CVPR 2023), frozen: its image encoder and its audio encoder, trained without the classes of the dataset | is the music of a class closer to the images of the same class than to the images of the other classes? |
| Image classifier | OpenCLIP ViT-H/14, the image encoder of ImageBind (kept frozen by ImageBind) | is each image class consistent? |
| Music classifier | CLAP for music and speech, `laion/larger_clap_music_and_speech` (Wu et al., ICASSP 2023) | is each music class consistent? |

## Results

Cross-modal test (all the files; chance level in brackets):

| Measure | Result |
|---|---:|
| Mean rank of the right image class among the 41, for each music class | 4.46 (21) |
| Music classes whose right image class is the nearest / in the 5 nearest | 14 / 30 of 41 (1 / 5) |
| Mean rank of the right music class, for each image class | 8.49 (21) |
| Music classes whose nearest image class is in the same group (historical, cultural, stylistic) | 33 of 41 (16.25) |
| Images of the same class among the 10 nearest to a track, mean of the classes | 17.16% (2.44%) |
| Tracks of the same class among the 10 nearest to an image, mean of the classes | 11.71% (2.44%) |

None of the 10000 permutations of the class labels (1000 for the nearest
files) reached these values (p = 0.0001 and p = 0.001).

Classifiers (test set; most frequent class in brackets for the accuracy,
chance 2.44% for the balanced accuracy):

| Classifier | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Images, 2574 images | 70.82% (8.43%) | 68.19% | 69.74% |
| Music (CLAP), 549 tracks | 79.23% (25.32%) | 45.31% | 43.28% |
| Music (ImageBind audio encoder, for comparison), 549 tracks | 78.51% (25.32%) | 43.52% | 42.13% |

The music classifiers miss most of the tracks of the classes with few tracks
(11 music classes have a single track in the test set); 82.5% of the errors
of the CLAP classifier fall in a class of the same group.

All the numbers are in `results/results.txt`, the values per class in
`results/classes.csv`, the 41 x 41 table of the cross-modal test (music
classes in rows, image classes in columns) in `results/similarity.csv`, the
confusion matrices of the test sets in `results/confusion_*.csv`.

## Requirements

- Python 3.12 and an NVIDIA GPU with CUDA (8 GB are enough);
- PyTorch with CUDA, then the other packages:
  ```
  pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```
- `ffmpeg` on the PATH;
- the ImageBind code at commit `53680b02d7e37b19b124fa37bae4b6c98c38f5be`
  and its weights (4.8 GB, SHA-256 `d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157`):
  ```
  git clone https://github.com/facebookresearch/ImageBind.git
  git -C ImageBind checkout 53680b02d7e37b19b124fa37bae4b6c98c38f5be
  curl -L -o imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
  ```
  Only the model files of ImageBind are used (pytorchvideo is not needed);
- the CLAP checkpoint is downloaded by `transformers` at its first use
  (revision `195c3a3e68faebb3e2088b9a79e79b43ddbda76b`, about 780 MB);
- the dataset from Kaggle (`Entire_images/`, `Entire_music/`,
  `metadata_images.csv`, `metadata_music.csv`).

Tested on Windows 11 with Python 3.12.10, torch 2.11.0+cu128, transformers
5.20.2, scikit-learn 1.9.1, an ffmpeg build of March 2026 and an NVIDIA
GeForce RTX 5050 Laptop GPU (8 GB).

## Usage

```
python 0_check_models.py   --imagebind ImageBind --checkpoint imagebind_huge.pth
python 1_image_vectors.py  --data path/to/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
python 2_music_vectors.py  --data path/to/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
python 3_clap_vectors.py   --data path/to/Museart_Large
python 4_analysis.py
```

| Option | Meaning |
|---|---|
| `--data` | the dataset as downloaded from Kaggle (default: `Museart_Large`) |
| `--output` | folder of the vectors, logs and results (default: `output/` next to the scripts) |
| `--imagebind` | the clone of the ImageBind repository (default: `ImageBind`) |
| `--checkpoint` | `imagebind_huge.pth` (default: `imagebind_huge.pth`) |

The defaults are in the CONFIG of `common.py`. Times on the GPU above: about
10 minutes (images), 30 minutes (ImageBind, music), 50 minutes (CLAP, music;
most of the time goes in decoding the audio from disk), 4 minutes (analysis,
on the CPU). Scripts 1-3 save their work in chunks and continue where they
stopped when they are run again.

## What the scripts do

0. `0_check_models.py` runs the two models on the demo files of ImageBind
   (a dog, a car, a bird): the vision x audio matrix must be close to the one
   of the ImageBind README, `[[0.807 0.109 0.084] [0.104 0.788 0.108] [0.002 0.002 0.996]]`
   (here `[[0.803 0.109 0.088] [0.099 0.796 0.105] [0.002 0.002 0.996]]`,
   with the audio decoded by ffmpeg), and CLAP puts 2 of the 3 sounds on the
   right text prompt.
1. `1_image_vectors.py`: one vector per image, preprocessed as in ImageBind
   (short side resized to 224, center crop 224).
2. `2_music_vectors.py`: the pieces of music, 10 s long, one every 2 minutes
   of music and at least 3 per track, centred in equal parts of the track
   (durations in `track_durations.csv`, measured by decoding every file):
   49847 pieces. Each piece is decoded from the first channel at 16 kHz and
   embedded as in ImageBind (3 clips of 2 s, mean of their vectors). Pieces
   whose mean amplitude is below -60 dBFS (as in `preprocess_example.py`),
   357, and pieces that cannot be decoded, 9, are left out: 49481 pieces
   have a vector. A track is the mean of its pieces.
3. `3_clap_vectors.py`: the same pieces, decoded at 48 kHz, through CLAP and
   its own feature extractor.
4. `4_analysis.py`:
   - cross-modal test: the centroid of each image class and of each music
     class; the cosine of every music class with every image class; the rank
     of the right class; a permutation test on the labels of the 41 image
     centroids (10000 permutations); for each track the 10 nearest images,
     and for each image the 10 nearest tracks, with a permutation test on the
     labels (1000 permutations);
   - classifiers: logistic regression on standardized vectors, trained on
     `train`, regularization chosen on the macro F1 of `val` among
     0.001-1, results on `test`; the music classifiers are trained on pieces
     and a track is predicted by the mean probability of its pieces.

`Entire_music/train/Late_romanticism_chamber_23/230013.m4a`, whose AAC data
are damaged (see the main README), gives no piece: the analysis uses 5413
tracks.

## Notes

- The CLAP checkpoint for music only, `laion/larger_clap_music` (revision
  `a0b4534`), is not used: its published weights are untrained (274 of its
  555 weight tensors are all zero) and it gives the same vector to every
  sound and every text.
- Licences of the models: ImageBind code and weights, CC BY-NC-SA 4.0
  (`LICENSE` of its repository); `laion/larger_clap_music_and_speech`,
  Apache-2.0.
- The vectors (about 400 MB) are not in the repository; the scripts compute
  them again. Run again on the same machine, the scripts gave the same
  vectors bit for bit (all the images; the first 100 tracks for the music).
  On another GPU or driver the last digits may change, since ImageBind runs
  in float16.
