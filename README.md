# Museart_Large

**A refined music-image dataset extending Museart (https://www.kaggle.com/datasets/alfredobaione/museart, https://doi.org/10.1145/3750069.3755967)**

Museart_Large is a music-image dataset of **5413 tracks** (1543.60 hours) and
**25683 images** of artworks, organized in 41 classes: historical periods,
geographical areas and musical genres, from *Pre-renaissance sacred* to
*Heavy metal*. Every track and every image comes with its vectors (ImageBind
and, for the music, CLAP) and its metadata; the **14546** images of works in
the public domain are also given as image files. Each class is split into
training, validation and test sets (80% / 10% / 10%).

| | Where |
|---|---|
| Data | Kaggle: **https://www.kaggle.com/datasets/alfredobaione/museart-large** |
| Code (technical validation) | GitHub: https://github.com/AlfredoBaione/Museart_large |
| Paper | Museart: A Refined Music-Image Dataset (CHItaly '25), https://doi.org/10.1145/3750069.3755967; see [Citation](#citation) |

## Contents

```
Entire_images/
    train/                        41 class folders: 20546 .npz files, 11650 image files
    val/                          41 class folders: 2563 .npz files, 1451 image files
    test/                         41 class folders: 2574 .npz files, 1445 image files
Entire_music/
    train/                        41 class folders: 4322 .npz files
    val/                          41 class folders: 542 .npz files
    test/                         41 class folders: 549 .npz files
metadata_images.csv               one row per image
metadata_music.csv                one row per track
```

Every image and every track has a `.npz` file with its vectors. The 14546
images of works in the public domain also have their image file, the
original one. The GitHub repository has the code, the two CSV files and one
example image for each class and set that has image files (94 images).

### File names

The `file` column of the CSV files gives the path of every track and image:
`<Entire_music or Entire_images>/<set>/<class folder>/<number>.<extension>`.
The vectors are in the file with the same path and the extension `.npz`:

```
Entire_music/val/Rock_35/350014.mp3                track (metadata_music.csv)
Entire_music/val/Rock_35/350014.npz                its vectors
Entire_images/train/Baroque_sacred_06/060769.jpg   image file
Entire_images/train/Baroque_sacred_06/060769.npz   its vectors
```

Each class has a two-digit code, `00` to `40` (classes in alphabetical
order), and a folder named after it: the class name with spaces replaced by
`_`, commas removed, and the code at the end (`Rock_35`, `Rap_hip_hop_32`). A
file number is the class code followed by four digits; in every class the
tracks, and separately the images, are numbered from `0001`.

## The 41 classes

In the historical classes, a track belongs to the period in which its work
was composed, and an image to the period in which its artist mainly worked.
Periods: Pre-renaissance up to 1419; Renaissance 1420–1599 (Flemish:
composers and painters of the Low Countries); Baroque 1600–1749; Classical
1750–1819 (painting: up to 1799); Romanticism 1820–1849 (painting: from
1800); Late romanticism 1850–1899; Early 20th century 1900–1929; 20th century
from 1930.

Hours: duration of the tracks. Image files: images also given as files.

| Code | Class | Folder | Tracks | Hours | Images | Image files |
|---:|---|---|---:|---:|---:|---:|
| 00 | 20th century experimental chamber | `20th_century_experimental_chamber_00` | 112 | 50.18 | 1146 | 79 |
| 01 | 20th century experimental orchestral | `20th_century_experimental_orchestral_01` | 114 | 59.31 | 1038 | 83 |
| 02 | 20th century traditional chamber | `20th_century_traditional_chamber_02` | 27 | 10.19 | 451 | 135 |
| 03 | 20th century traditional orchestral | `20th_century_traditional_orchestral_03` | 38 | 23.86 | 397 | 25 |
| 04 | African traditional folklore | `African_traditional_folklore_04` | 18 | 13.83 | 1102 | 69 |
| 05 | Afro-American | `Afro-American_05` | 81 | 14.51 | 743 | 89 |
| 06 | Baroque sacred | `Baroque_sacred_06` | 97 | 23.64 | 843 | 840 |
| 07 | Baroque secular | `Baroque_secular_07` | 1238 | 88.52 | 939 | 939 |
| 08 | Classical sacred | `Classical_sacred_08` | 3 | 4.34 | 370 | 370 |
| 09 | Classical secular | `Classical_secular_09` | 51 | 22.98 | 335 | 335 |
| 10 | Country | `Country_10` | 11 | 14.19 | 111 | 8 |
| 11 | Dance | `Dance_11` | 345 | 110.23 | 214 | 0 |
| 12 | Early 20th century experimental chamber | `Early_20th_century_experimental_chamber_12` | 25 | 16.18 | 681 | 678 |
| 13 | Early 20th century experimental orchestral | `Early_20th_century_experimental_orchestral_13` | 21 | 10.07 | 748 | 748 |
| 14 | Early 20th century traditional chamber | `Early_20th_century_traditional_chamber_14` | 40 | 23.87 | 356 | 356 |
| 15 | Early 20th century traditional orchestral | `Early_20th_century_traditional_orchestral_15` | 35 | 27.97 | 721 | 721 |
| 16 | East Asian traditional folklore | `East_Asian_traditional_folklore_16` | 18 | 18.76 | 870 | 568 |
| 17 | East European folklore | `East_European_folklore_17` | 25 | 19.13 | 534 | 210 |
| 18 | Electronic | `Electronic_18` | 78 | 38.28 | 745 | 0 |
| 19 | Flemish sacred | `Flemish_sacred_19` | 5 | 3.98 | 205 | 205 |
| 20 | Flemish secular | `Flemish_secular_20` | 9 | 6.19 | 212 | 212 |
| 21 | Heavy metal | `Heavy_metal_21` | 105 | 64.17 | 1330 | 0 |
| 22 | Jazz | `Jazz_22` | 1389 | 207.02 | 303 | 0 |
| 23 | Late romanticism chamber | `Late_romanticism_chamber_23` | 82 | 38.23 | 1987 | 1985 |
| 24 | Late romanticism orchestral | `Late_romanticism_orchestral_24` | 42 | 45.40 | 2170 | 2170 |
| 25 | Latin-American | `Latin-American_25` | 30 | 28.91 | 958 | 213 |
| 26 | Middle, north European folklore | `Middle_north_European_folklore_26` | 40 | 10.09 | 431 | 261 |
| 27 | Middle, south Asian traditional folklore | `Middle_south_Asian_traditional_folklore_27` | 21 | 21.35 | 484 | 85 |
| 28 | North American traditional folklore | `North_American_traditional_folklore_28` | 9 | 4.50 | 431 | 0 |
| 29 | Pop | `Pop_29` | 47 | 58.54 | 509 | 0 |
| 30 | Pre-renaissance sacred | `Pre-renaissance_sacred_30` | 3 | 3.80 | 408 | 407 |
| 31 | Pre-renaissance secular | `Pre-renaissance_secular_31` | 7 | 7.16 | 180 | 177 |
| 32 | Rap, hip hop | `Rap_hip_hop_32` | 361 | 82.81 | 132 | 0 |
| 33 | Renaissance sacred | `Renaissance_sacred_33` | 6 | 6.53 | 696 | 696 |
| 34 | Renaissance secular | `Renaissance_secular_34` | 6 | 6.23 | 1029 | 1025 |
| 35 | Rock | `Rock_35` | 508 | 163.22 | 258 | 0 |
| 36 | Romanticism chamber | `Romanticism_chamber_36` | 70 | 26.49 | 489 | 489 |
| 37 | Romanticism orchestral | `Romanticism_orchestral_37` | 12 | 10.46 | 217 | 217 |
| 38 | South American traditional folklore | `South_American_traditional_folklore_38` | 9 | 6.95 | 440 | 39 |
| 39 | South European folklore | `South_European_folklore_39` | 24 | 20.28 | 295 | 112 |
| 40 | Western songwriting | `Western_songwriting_40` | 251 | 131.26 | 175 | 0 |
| | **Total** | | **5413** | **1543.60** | **25683** | **14546** |

## Training, validation and test sets

Every class is split into the three sets with 80% / 10% / 10% of its tracks
and of its images (largest remainder; every class has at least one track and
one image in each set), in a random order with a fixed seed (42). Files with
the same content are in the same set: identical images, reproductions of the
same artwork, and tracks that share their audio, such as the same recording
in two classes or a track contained in a whole album.

| Set | Tracks | Hours | Images | Image files |
|---|---:|---:|---:|---:|
| training (`train`) | 4322 (79.8%) | 1217.66 (78.9%) | 20546 (80.0%) | 11650 |
| validation (`val`) | 542 (10.0%) | 151.99 (9.8%) | 2563 (10.0%) | 1451 |
| test (`test`) | 549 (10.1%) | 173.95 (11.3%) | 2574 (10.0%) | 1445 |
| total | 5413 | 1543.60 | 25683 | 14546 |

Per class (training / validation / test):

| Code | Folder | Tracks | Hours | Images |
|---:|---|---:|---:|---:|
| 00 | `20th_century_experimental_chamber_00` | 90 / 11 / 11 | 39.16 / 4.55 / 6.47 | 917 / 114 / 115 |
| 01 | `20th_century_experimental_orchestral_01` | 91 / 11 / 12 | 49.20 / 6.09 / 4.02 | 830 / 104 / 104 |
| 02 | `20th_century_traditional_chamber_02` | 21 / 3 / 3 | 8.28 / 1.83 / 0.08 | 361 / 45 / 45 |
| 03 | `20th_century_traditional_orchestral_03` | 30 / 4 / 4 | 20.70 / 1.88 / 1.28 | 317 / 40 / 40 |
| 04 | `African_traditional_folklore_04` | 14 / 2 / 2 | 11.90 / 1.22 / 0.72 | 882 / 110 / 110 |
| 05 | `Afro-American_05` | 65 / 8 / 8 | 9.98 / 0.68 / 3.85 | 595 / 74 / 74 |
| 06 | `Baroque_sacred_06` | 77 / 10 / 10 | 17.47 / 1.61 / 4.56 | 675 / 84 / 84 |
| 07 | `Baroque_secular_07` | 990 / 124 / 124 | 70.92 / 8.02 / 9.58 | 751 / 94 / 94 |
| 08 | `Classical_sacred_08` | 1 / 1 / 1 | 1.72 / 0.92 / 1.70 | 296 / 37 / 37 |
| 09 | `Classical_secular_09` | 41 / 5 / 5 | 18.47 / 0.89 / 3.63 | 268 / 33 / 34 |
| 10 | `Country_10` | 9 / 1 / 1 | 11.10 / 1.50 / 1.59 | 89 / 11 / 11 |
| 11 | `Dance_11` | 276 / 34 / 35 | 79.72 / 19.57 / 10.95 | 171 / 21 / 22 |
| 12 | `Early_20th_century_experimental_chamber_12` | 20 / 2 / 3 | 12.05 / 1.28 / 2.85 | 545 / 68 / 68 |
| 13 | `Early_20th_century_experimental_orchestral_13` | 17 / 2 / 2 | 8.99 / 0.79 / 0.29 | 598 / 75 / 75 |
| 14 | `Early_20th_century_traditional_chamber_14` | 32 / 4 / 4 | 20.34 / 1.45 / 2.08 | 285 / 35 / 36 |
| 15 | `Early_20th_century_traditional_orchestral_15` | 28 / 3 / 4 | 20.72 / 4.65 / 2.60 | 577 / 72 / 72 |
| 16 | `East_Asian_traditional_folklore_16` | 14 / 2 / 2 | 13.81 / 2.41 / 2.54 | 696 / 87 / 87 |
| 17 | `East_European_folklore_17` | 20 / 2 / 3 | 14.72 / 1.53 / 2.87 | 427 / 53 / 54 |
| 18 | `Electronic_18` | 62 / 8 / 8 | 28.79 / 6.32 / 3.17 | 596 / 74 / 75 |
| 19 | `Flemish_sacred_19` | 3 / 1 / 1 | 2.62 / 0.52 / 0.84 | 164 / 20 / 21 |
| 20 | `Flemish_secular_20` | 7 / 1 / 1 | 4.88 / 1.23 / 0.08 | 170 / 21 / 21 |
| 21 | `Heavy_metal_21` | 84 / 10 / 11 | 53.09 / 4.54 / 6.53 | 1064 / 133 / 133 |
| 22 | `Jazz_22` | 1111 / 139 / 139 | 168.51 / 17.26 / 21.25 | 243 / 30 / 30 |
| 23 | `Late_romanticism_chamber_23` | 66 / 8 / 8 | 32.49 / 1.77 / 3.98 | 1589 / 199 / 199 |
| 24 | `Late_romanticism_orchestral_24` | 34 / 4 / 4 | 40.20 / 2.45 / 2.75 | 1736 / 217 / 217 |
| 25 | `Latin-American_25` | 24 / 3 / 3 | 22.96 / 3.48 / 2.47 | 766 / 96 / 96 |
| 26 | `Middle_north_European_folklore_26` | 32 / 4 / 4 | 7.22 / 1.51 / 1.37 | 345 / 43 / 43 |
| 27 | `Middle_south_Asian_traditional_folklore_27` | 17 / 2 / 2 | 17.13 / 1.82 / 2.40 | 387 / 48 / 49 |
| 28 | `North_American_traditional_folklore_28` | 7 / 1 / 1 | 1.33 / 2.18 / 0.99 | 345 / 43 / 43 |
| 29 | `Pop_29` | 37 / 5 / 5 | 52.59 / 1.12 / 4.83 | 407 / 51 / 51 |
| 30 | `Pre-renaissance_sacred_30` | 1 / 1 / 1 | 0.91 / 1.17 / 1.72 | 326 / 41 / 41 |
| 31 | `Pre-renaissance_secular_31` | 5 / 1 / 1 | 5.35 / 0.71 / 1.10 | 144 / 18 / 18 |
| 32 | `Rap_hip_hop_32` | 289 / 36 / 36 | 68.56 / 8.05 / 6.20 | 106 / 13 / 13 |
| 33 | `Renaissance_sacred_33` | 4 / 1 / 1 | 4.38 / 0.91 / 1.23 | 557 / 69 / 70 |
| 34 | `Renaissance_secular_34` | 4 / 1 / 1 | 3.75 / 1.88 / 0.60 | 823 / 103 / 103 |
| 35 | `Rock_35` | 406 / 51 / 51 | 120.74 / 15.57 / 26.91 | 206 / 26 / 26 |
| 36 | `Romanticism_chamber_36` | 56 / 7 / 7 | 20.02 / 0.79 / 5.68 | 391 / 49 / 49 |
| 37 | `Romanticism_orchestral_37` | 10 / 1 / 1 | 8.28 / 1.12 / 1.06 | 173 / 22 / 22 |
| 38 | `South_American_traditional_folklore_38` | 7 / 1 / 1 | 5.44 / 0.62 / 0.89 | 352 / 44 / 44 |
| 39 | `South_European_folklore_39` | 19 / 2 / 3 | 14.88 / 2.66 / 2.74 | 236 / 29 / 30 |
| 40 | `Western_songwriting_40` | 201 / 25 / 25 | 104.31 / 13.44 / 13.52 | 140 / 17 / 18 |

## Metadata

UTF-8 CSV files, comma-separated, with Windows line ends (CRLF).

`metadata_music.csv`, one row per track:

| Column | Content |
|---|---|
| `file` | path of the track (see [File names](#file-names)) |
| `class` | class name |
| `split` | set: `train`, `val` or `test` |
| `info` | original file name of the track, tidied |
| `title`, `artist`, `album`, `date`, `composer` | from the tags of the original file, cleaned and completed with the information of its name; in classical music `artist` can hold the performer or the composer |

Filled values: `info` 5241, `title` 5170, `artist` 4267, `album` 3404, `date`
2067, `composer` 1569 (of 5413 tracks).

```
file,class,split,info,title,artist,album,date,composer
Entire_music/val/Rock_35/350014.mp3,Rock,val,01-Hells Bells,Hells Bells,AC/DC,Back In Black,1980,
Entire_music/val/Baroque_sacred_06/060001.mp3,Baroque sacred,val,"01 - Prelude in E Flat Major, BWV 552","Prelude in E Flat Major, BWV 552",Wolfgang Ruebsam,"Clavierübung III, Vol.1 - J.S.Bach",1994,Johann Sebastian Bach
```

`metadata_images.csv`, one row per image:

| Column | Content |
|---|---|
| `file` | path of the image (see [File names](#file-names)) |
| `class` | class name |
| `split` | set: `train`, `val` or `test` |
| `artist`, `title`, `year` | from the original file name, where it gives them; names of works, manuscripts and categories given in place of the artist (e.g. *Codex Manesse*, *Ancient Egipt*) are in `title` |

Filled values: `artist` 23320, `title` 5994, `year` 3306 (of 25683 images);
1673 images have none.

```
file,class,split,artist,title,year
Entire_images/train/Baroque_sacred_06/060769.jpg,Baroque sacred,train,Jacob Jordaens,Bust of Satyr,1621
Entire_images/val/Pop_29/290001.jpg,Pop,val,Andy Warhol,,
```

## Vectors

The vectors are those of the technical validation (code in
`technical_validation/` of the GitHub repository): float32, L2-normalized,
NumPy arrays in `.npz` files.

- **Images**: array `imagebind` (1024 values), from the vision encoder of
  ImageBind huge (OpenCLIP ViT-H/14); the image is resized to 224 pixels on
  its shorter side (bicubic), center-cropped to 224 × 224 and normalized as
  in CLIP.
- **Music**: one row per piece of 10 s. The pieces are one for every
  2 minutes of music, at least 3 per track, centered in equal parts of the
  track, decodable and not silent (mean amplitude of -60 dBFS or more):
  49481 pieces.
  - `imagebind` (pieces × 1024): audio encoder of ImageBind huge, in the same
    space as the image vectors; first channel at 16 kHz, 3 clips of 2 s, each
    turned into a 128-bin filterbank as in ImageBind, mean of the 3 clip
    vectors;
  - `clap` (pieces × 512): audio embedding of
    `laion/larger_clap_music_and_speech` (revision `195c3a3`), first channel
    at 48 kHz;
  - `start_s`: start of each piece, in seconds.

```python
import csv
import numpy as np

rows = list(csv.DictReader(open('metadata_images.csv', encoding='utf-8')))
images = np.stack([np.load(r['file'].rsplit('.', 1)[0] + '.npz')['imagebind'] for r in rows])
track = np.load('Entire_music/val/Rock_35/350014.npz')

vector = track['imagebind'].mean(axis=0)
nearest = np.argsort(images @ vector)[::-1][:10]
print([rows[i]['artist'] for i in nearest])
```

## Duplicates

Within a class, image files are unique (SHA-1) and each recording appears
once as a track (acoustic fingerprint, Haitsma & Kalker, 2002), although a
track can also be contained in a longer file, such as a whole album. The same
image, or the same track, can belong to two classes, as a file of each.

## Copyright

Image files are given for the works whose authors died in 1955 or earlier. In
the European Union a work is protected for the life of its author and
70 years after the death, counted from 1 January of the following year
(Directive 2006/116/EC, art. 1 and 8), and faithful reproductions of works of
visual art whose protection has expired are not protected (Directive (EU)
2019/790, art. 14). The author of each image is the one named in `artist`,
with the date of death from Wikidata, checked by hand for ambiguous names; old
works without a named author, such as Fayum portraits and medieval manuscripts
and frescoes (named in `title`), are given as files. Images
whose author died in 1956 or later or, in the classes from 1850 on, is not
identified, images of several authors and photographs of three-dimensional
works (sculptures, objects, buildings) are given as vectors, as is all the
music.

## License

- Code: MIT License, see `LICENSE` in the GitHub repository.
- Data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

**Alfredo Baione and Genoveffa Tortora. 2025. Museart: A Refined Music-Image Dataset. In Proceedings of the 16th Biannual Conference of the Italian SIGCHI Chapter (CHItaly '25). Association for Computing Machinery, New York, NY, USA, Article 107, 1–3. https://doi.org/10.1145/3750069.3755967**
