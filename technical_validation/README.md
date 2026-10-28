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
| Mean rank of the right image class among the 41, for each music class | 4.54 (21) |
| Music classes whose right image class is the nearest / in the 5 nearest | 15 / 31 of 41 (1 / 5) |
| Mean rank of the right music class, for each image class | 8.32 (21) |
| Music classes whose nearest image class is in the same group (historical, geographical, stylistic) | 35 of 41 (16.30) |
| Images of the same class among the 10 nearest to a track, mean of the classes | 17.52% (2.44%) |
| Tracks of the same class among the 10 nearest to an image, mean of the classes | 11.89% (2.44%) |

None of the 10000 permutations of the class labels (1000 for the nearest
files) reached these values (p = 0.0001 and p = 0.001).

Classifiers (test set; most frequent class in brackets for the accuracy,
chance 2.44% for the balanced accuracy):

| Classifier | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Images, 2574 images | 70.82% (8.43%) | 68.19% | 69.74% |
| Music (CLAP), 547 tracks | 79.52% (25.41%) | 40.30% | 37.92% |
| Music (ImageBind audio encoder, for comparison), 547 tracks | 82.08% (25.41%) | 54.25% | 51.84% |

The music classifiers miss most of the tracks of the classes with few tracks
(11 music classes have a single track in the test set); 84.8% of the errors
of the CLAP classifier fall in a class of the same group.

All the numbers are in `results/results.txt`, the values per class in
`results/classes.csv`, the 41 x 41 table of the cross-modal test (music
classes in rows, image classes in columns) in `results/similarity.csv`, the
confusion matrices of the test sets in `results/confusion_*.csv`.

## Run the analysis on the published vectors

The vectors are published with the dataset on Kaggle: the `ib_` file of every
image and the `ib_` and `c_` files of the pieces of every segment (see the
main README). With Python 3.9 or later, numpy and scikit-learn
(`pip install numpy scikit-learn`):

```
python dataset_vectors.py --data path/to/Museart_Large
python 3_analysis.py
```

`dataset_vectors.py` writes the vectors in `output/` in the files read by
`3_analysis.py`, which gives the results above in about 50 minutes on the CPU.

## Compute the vectors again

Scripts 0-2 compute the vectors from the original image and audio files.

### Requirements

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
- the CLAP checkpoint is downloaded by `transformers` at its first use
  (revision `195c3a3e68faebb3e2088b9a79e79b43ddbda76b`, about 780 MB);
- the original image and audio files, in the folders of the dataset.

Tested on Windows 11 with Python 3.12.10, torch 2.11.0+cu128, transformers
5.20.2, scikit-learn 1.9.1, an ffmpeg build of March 2026 and an NVIDIA
GeForce RTX 5050 Laptop GPU (8 GB).

### Usage

```
python 0_check_models.py   --imagebind ImageBind --checkpoint imagebind_huge.pth
python 1_image_vectors.py  --data path/to/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
python 2_music_vectors.py  --data path/to/Museart_Large --imagebind ImageBind --checkpoint imagebind_huge.pth
python 3_analysis.py
```

| Option | Meaning |
|---|---|
| `--data` | the folder of the dataset (default: `Museart_Large`) |
| `--output` | folder of the vectors, logs and results (default: `output/` next to the scripts) |
| `--imagebind` | the clone of the ImageBind repository (default: `ImageBind`) |
| `--checkpoint` | `imagebind_huge.pth` (default: `imagebind_huge.pth`) |

The defaults are at the top of `common.py`. Times on the GPU above: about
10 minutes (images), 4 hours and 20 minutes (music, both models), 50 minutes
(analysis, on the CPU). Scripts 1 and 2 save their work as they go (chunks of
images, one file per track) and continue where they stopped when they are run
again.

## What the scripts do

0. `0_check_models.py` runs the two models on the demo files of ImageBind
   (a dog, a car, a bird): the vision x audio matrix must be close to the one
   of the ImageBind README, `[[0.807 0.109 0.084] [0.104 0.788 0.108] [0.002 0.002 0.996]]`
   (here `[[0.803 0.109 0.088] [0.099 0.796 0.105] [0.002 0.002 0.996]]`,
   with the audio decoded by ffmpeg), and CLAP puts 2 of the 3 sounds on the
   right text prompt.
1. `1_image_vectors.py`: one vector per image, preprocessed as in ImageBind
   (short side resized to 224, center crop 224).
2. `2_music_vectors.py`: every track is decoded once from its start (first
   channel, at 16 kHz and at 48 kHz) and read in segments of 30 s; each
   segment of `metadata_music.csv` gives three pieces of 10 s, and a piece
   whose mean amplitude is below -60 dBFS gets no vector: 542829 pieces.
   ImageBind embeds a piece as the mean of the vectors of 3 clips of 2 s;
   CLAP embeds it with the log-mel features of its feature extractor,
   computed on the GPU. A track is the mean of its pieces.
3. `3_analysis.py`:
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

## Notes

- Licences of the models: ImageBind code and weights, CC BY-NC-SA 4.0
  (`LICENSE` of its repository); `laion/larger_clap_music_and_speech`,
  Apache-2.0.
- Run again on the same machine, scripts 1 and 2 gave the published vectors
  bit for bit (all the images; for the music, 11 tracks of all the formats).
  The music vectors can differ by up to 2e-4 when the pieces are embedded in
  batches of a different composition, and on another GPU or driver the last
  digits may change, since ImageBind runs in float16.
