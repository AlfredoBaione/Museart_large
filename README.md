# Museart_Large

**A refined music-image dataset extending Museart (https://www.kaggle.com/datasets/alfredobaione/museart, https://doi.org/10.1145/3750069.3755967)**

Museart_Large is a music-image dataset of **181352 music segments** of 30
seconds (1511.27 hours, from 5399 tracks) and **25683 images** of artworks,
organized in 41 classes: historical periods, geographical areas and musical
genres, from *Pre-renaissance sacred* to *Heavy metal*. Every segment and
every image comes with its vectors (ImageBind and, for the music, CLAP) and
its metadata; the **14546** images of works in the public domain are also
given as image files. Each class is split into training, validation and test
sets (80% / 10% / 10%).

| | Where |
|---|---|
| Data | Kaggle: **https://www.kaggle.com/datasets/alfredobaione/museart-large** |
| Code (technical validation) | GitHub: https://github.com/AlfredoBaione/Museart_large |
| Paper | Museart: A Refined Music-Image Dataset (CHItaly '25), https://doi.org/10.1145/3750069.3755967; see [Citation](#citation) |

## Contents

```
Entire_images/
    train/                        41 class folders: 20546 ib files, 11650 image files
    val/                          41 class folders: 2563 ib files, 1451 image files
    test/                         41 class folders: 2574 ib files, 1445 image files
Entire_music/
    train/                        41 class folders: 428603 ib files, 428603 c files
    val/                          41 class folders: 52965 ib files, 52965 c files
    test/                         41 class folders: 61261 ib files, 61261 c files
metadata_images.csv               one row per image
metadata_music.csv                one row per segment
```

The vectors are in `.npz` files, one for each model: `ib_` for ImageBind and
`c_` for CLAP. Every image has its `ib_` file, and every segment an `ib_` and
a `c_` file for each of its pieces of 10 s that is not silent. The 14546
images of works in the public domain also have their image file, the
original one. The GitHub repository has the code, the two CSV files and one
example image for each class and set that has image files (94 images).

### File names

The `file` column of the CSV files gives the path of every image and
segment; its vector files are in the same folder. The pieces of a segment are
numbered 1, 2 and 3 (seconds 0-10, 10-20 and 20-30):

```
Entire_images/train/Baroque_sacred_06/060769.jpg      image (metadata_images.csv)
Entire_images/train/Baroque_sacred_06/ib_060769.npz   its ImageBind vector
Entire_music/val/Rock_35/350014_001                   segment 1 of track 350014 (metadata_music.csv)
Entire_music/val/Rock_35/ib_350014_001_1.npz          ImageBind vector of its piece 1
Entire_music/val/Rock_35/c_350014_001_1.npz           CLAP vector of its piece 1
```

Each class has a two-digit code, `00` to `40` (classes in alphabetical
order), and a folder named after it: the class name with spaces replaced by
`_`, commas removed, and the code at the end (`Rock_35`, `Rap_hip_hop_32`). A
track or image number is the class code followed by four digits; in every
class the tracks, and separately the images, are numbered from `0001`. The
segments of a track are numbered from `001`: segment *n* covers the seconds
from 30(*n* - 1) to 30*n* of the track.

## The 41 classes

In the historical classes, a track belongs to the period in which its work
was composed, and an image to the period in which its artist mainly worked.
Periods: Pre-renaissance up to 1419; Renaissance 1420–1599 (Flemish:
composers and painters of the Low Countries); Baroque 1600–1749; Classical
1750–1819 (painting: up to 1799); Romanticism 1820–1849 (painting: from
1800); Late romanticism 1850–1899; Early 20th century 1900–1929; 20th century
from 1930.

Hours: duration of the segments. Image files: images also given as files.

| Code | Class | Folder | Tracks | Segments | Hours | Images | Image files |
|---:|---|---|---:|---:|---:|---:|---:|
| 00 | 20th century experimental chamber | `20th_century_experimental_chamber_00` | 112 | 5945 | 49.54 | 1146 | 79 |
| 01 | 20th century experimental orchestral | `20th_century_experimental_orchestral_01` | 114 | 7055 | 58.79 | 1038 | 83 |
| 02 | 20th century traditional chamber | `20th_century_traditional_chamber_02` | 27 | 1204 | 10.03 | 451 | 135 |
| 03 | 20th century traditional orchestral | `20th_century_traditional_orchestral_03` | 38 | 2843 | 23.69 | 397 | 25 |
| 04 | African traditional folklore | `African_traditional_folklore_04` | 18 | 1649 | 13.74 | 1102 | 69 |
| 05 | Afro-American | `Afro-American_05` | 81 | 1703 | 14.19 | 743 | 89 |
| 06 | Baroque sacred | `Baroque_sacred_06` | 97 | 2794 | 23.28 | 843 | 840 |
| 07 | Baroque secular | `Baroque_secular_07` | 1236 | 10022 | 83.52 | 939 | 939 |
| 08 | Classical sacred | `Classical_sacred_08` | 3 | 519 | 4.33 | 370 | 370 |
| 09 | Classical secular | `Classical_secular_09` | 51 | 2730 | 22.75 | 335 | 335 |
| 10 | Country | `Country_10` | 11 | 1319 | 10.99 | 111 | 8 |
| 11 | Dance | `Dance_11` | 343 | 13019 | 108.49 | 214 | 0 |
| 12 | Early 20th century experimental chamber | `Early_20th_century_experimental_chamber_12` | 25 | 1930 | 16.08 | 681 | 678 |
| 13 | Early 20th century experimental orchestral | `Early_20th_century_experimental_orchestral_13` | 21 | 1196 | 9.97 | 748 | 748 |
| 14 | Early 20th century traditional chamber | `Early_20th_century_traditional_chamber_14` | 40 | 2847 | 23.73 | 356 | 356 |
| 15 | Early 20th century traditional orchestral | `Early_20th_century_traditional_orchestral_15` | 35 | 3339 | 27.82 | 721 | 721 |
| 16 | East Asian traditional folklore | `East_Asian_traditional_folklore_16` | 18 | 2243 | 18.69 | 870 | 568 |
| 17 | East European folklore | `East_European_folklore_17` | 25 | 2283 | 19.02 | 534 | 210 |
| 18 | Electronic | `Electronic_18` | 78 | 4551 | 37.92 | 745 | 0 |
| 19 | Flemish sacred | `Flemish_sacred_19` | 5 | 475 | 3.96 | 205 | 205 |
| 20 | Flemish secular | `Flemish_secular_20` | 9 | 739 | 6.16 | 212 | 212 |
| 21 | Heavy metal | `Heavy_metal_21` | 105 | 7623 | 63.52 | 1330 | 0 |
| 22 | Jazz | `Jazz_22` | 1389 | 23924 | 199.37 | 303 | 0 |
| 23 | Late romanticism chamber | `Late_romanticism_chamber_23` | 76 | 4484 | 37.37 | 1987 | 1985 |
| 24 | Late romanticism orchestral | `Late_romanticism_orchestral_24` | 42 | 5429 | 45.24 | 2170 | 2170 |
| 25 | Latin-American | `Latin-American_25` | 30 | 3450 | 28.75 | 958 | 213 |
| 26 | Middle, north European folklore | `Middle_north_European_folklore_26` | 40 | 1190 | 9.92 | 431 | 261 |
| 27 | Middle, south Asian traditional folklore | `Middle_south_Asian_traditional_folklore_27` | 21 | 2549 | 21.24 | 484 | 85 |
| 28 | North American traditional folklore | `North_American_traditional_folklore_28` | 9 | 536 | 4.47 | 431 | 0 |
| 29 | Pop | `Pop_29` | 47 | 6982 | 58.18 | 509 | 0 |
| 30 | Pre-renaissance sacred | `Pre-renaissance_sacred_30` | 3 | 454 | 3.78 | 408 | 407 |
| 31 | Pre-renaissance secular | `Pre-renaissance_secular_31` | 7 | 855 | 7.12 | 180 | 177 |
| 32 | Rap, hip hop | `Rap_hip_hop_32` | 359 | 9643 | 80.36 | 132 | 0 |
| 33 | Renaissance sacred | `Renaissance_sacred_33` | 6 | 781 | 6.51 | 696 | 696 |
| 34 | Renaissance secular | `Renaissance_secular_34` | 6 | 745 | 6.21 | 1029 | 1025 |
| 35 | Rock | `Rock_35` | 508 | 19264 | 160.53 | 258 | 0 |
| 36 | Romanticism chamber | `Romanticism_chamber_36` | 68 | 3140 | 26.17 | 489 | 489 |
| 37 | Romanticism orchestral | `Romanticism_orchestral_37` | 12 | 1251 | 10.43 | 217 | 217 |
| 38 | South American traditional folklore | `South_American_traditional_folklore_38` | 9 | 829 | 6.91 | 440 | 39 |
| 39 | South European folklore | `South_European_folklore_39` | 24 | 2420 | 20.17 | 295 | 112 |
| 40 | Western songwriting | `Western_songwriting_40` | 251 | 15398 | 128.32 | 175 | 0 |
| | **Total** | | **5399** | **181352** | **1511.27** | **25683** | **14546** |

## Training, validation and test sets

Every class is split into the three sets with 80% / 10% / 10% of its tracks
and of its images (largest remainder; every class has at least one track and
one image in each set), in a random order with a fixed seed (42). Files with
the same content are in the same set: identical images, reproductions of the
same artwork, and tracks that share their audio, such as the same recording
in two classes or a track contained in a whole album. All the segments of a
track are in the set of the track. This avoids the leakage of Museart, where
64 groups of identical images (129 files) and the segments of 949 of its 1310
tracks are in different sets.

| Set | Tracks | Segments | Hours | Images | Image files |
|---|---:|---:|---:|---:|---:|
| training (`train`) | 4311 (79.8%) | 143200 (79.0%) | 1193.33 | 20546 (80.0%) | 11650 |
| validation (`val`) | 541 (10.0%) | 17699 (9.8%) | 147.49 | 2563 (10.0%) | 1451 |
| test (`test`) | 547 (10.1%) | 20453 (11.3%) | 170.44 | 2574 (10.0%) | 1445 |
| total | 5399 | 181352 | 1511.27 | 25683 | 14546 |

Per class (training / validation / test):

| Code | Folder | Tracks | Segments | Images |
|---:|---|---:|---:|---:|
| 00 | `20th_century_experimental_chamber_00` | 90 / 11 / 11 | 4645 / 535 / 765 | 917 / 114 / 115 |
| 01 | `20th_century_experimental_orchestral_01` | 91 / 11 / 12 | 5854 / 726 / 475 | 830 / 104 / 104 |
| 02 | `20th_century_traditional_chamber_02` | 21 / 3 / 3 | 979 / 218 / 7 | 361 / 45 / 45 |
| 03 | `20th_century_traditional_orchestral_03` | 30 / 4 / 4 | 2468 / 224 / 151 | 317 / 40 / 40 |
| 04 | `African_traditional_folklore_04` | 14 / 2 / 2 | 1420 / 144 / 85 | 882 / 110 / 110 |
| 05 | `Afro-American_05` | 65 / 8 / 8 | 1167 / 78 / 458 | 595 / 74 / 74 |
| 06 | `Baroque_sacred_06` | 77 / 10 / 10 | 2062 / 189 / 543 | 675 / 84 / 84 |
| 07 | `Baroque_secular_07` | 989 / 124 / 123 | 8026 / 904 / 1092 | 751 / 94 / 94 |
| 08 | `Classical_sacred_08` | 1 / 1 / 1 | 206 / 110 / 203 | 296 / 37 / 37 |
| 09 | `Classical_secular_09` | 41 / 5 / 5 | 2194 / 104 / 432 | 268 / 33 / 34 |
| 10 | `Country_10` | 9 / 1 / 1 | 1133 / 90 / 96 | 89 / 11 / 11 |
| 11 | `Dance_11` | 275 / 34 / 34 | 9397 / 2326 / 1296 | 171 / 21 / 22 |
| 12 | `Early_20th_century_experimental_chamber_12` | 20 / 2 / 3 | 1436 / 153 / 341 | 545 / 68 / 68 |
| 13 | `Early_20th_century_experimental_orchestral_13` | 17 / 2 / 2 | 1070 / 93 / 33 | 598 / 75 / 75 |
| 14 | `Early_20th_century_traditional_chamber_14` | 32 / 4 / 4 | 2426 / 173 / 248 | 285 / 35 / 36 |
| 15 | `Early_20th_century_traditional_orchestral_15` | 28 / 3 / 4 | 2473 / 557 / 309 | 577 / 72 / 72 |
| 16 | `East_Asian_traditional_folklore_16` | 14 / 2 / 2 | 1652 / 288 / 303 | 696 / 87 / 87 |
| 17 | `East_European_folklore_17` | 20 / 2 / 3 | 1757 / 183 / 343 | 427 / 53 / 54 |
| 18 | `Electronic_18` | 62 / 8 / 8 | 3420 / 755 / 376 | 596 / 74 / 75 |
| 19 | `Flemish_sacred_19` | 3 / 1 / 1 | 312 / 62 / 101 | 164 / 20 / 21 |
| 20 | `Flemish_secular_20` | 7 / 1 / 1 | 582 / 147 / 10 | 170 / 21 / 21 |
| 21 | `Heavy_metal_21` | 84 / 10 / 11 | 6303 / 541 / 779 | 1064 / 133 / 133 |
| 22 | `Jazz_22` | 1111 / 139 / 139 | 19452 / 1995 / 2477 | 243 / 30 / 30 |
| 23 | `Late_romanticism_chamber_23` | 61 / 7 / 8 | 3813 / 199 / 472 | 1589 / 199 / 199 |
| 24 | `Late_romanticism_orchestral_24` | 34 / 4 / 4 | 4809 / 293 / 327 | 1736 / 217 / 217 |
| 25 | `Latin-American_25` | 24 / 3 / 3 | 2739 / 417 / 294 | 766 / 96 / 96 |
| 26 | `Middle_north_European_folklore_26` | 32 / 4 / 4 | 849 / 179 / 162 | 345 / 43 / 43 |
| 27 | `Middle_south_Asian_traditional_folklore_27` | 17 / 2 / 2 | 2044 / 217 / 288 | 387 / 48 / 49 |
| 28 | `North_American_traditional_folklore_28` | 7 / 1 / 1 | 157 / 261 / 118 | 345 / 43 / 43 |
| 29 | `Pop_29` | 37 / 5 / 5 | 6273 / 133 / 576 | 407 / 51 / 51 |
| 30 | `Pre-renaissance_sacred_30` | 1 / 1 / 1 | 108 / 140 / 206 | 326 / 41 / 41 |
| 31 | `Pre-renaissance_secular_31` | 5 / 1 / 1 | 640 / 84 / 131 | 144 / 18 / 18 |
| 32 | `Rap_hip_hop_32` | 287 / 36 / 36 | 7970 / 946 / 727 | 106 / 13 / 13 |
| 33 | `Renaissance_sacred_33` | 4 / 1 / 1 | 524 / 109 / 148 | 557 / 69 / 70 |
| 34 | `Renaissance_secular_34` | 4 / 1 / 1 | 449 / 225 / 71 | 823 / 103 / 103 |
| 35 | `Rock_35` | 406 / 51 / 51 | 14275 / 1826 / 3163 | 206 / 26 / 26 |
| 36 | `Romanticism_chamber_36` | 54 / 7 / 7 | 2371 / 91 / 678 | 391 / 49 / 49 |
| 37 | `Romanticism_orchestral_37` | 10 / 1 / 1 | 990 / 134 / 127 | 173 / 22 / 22 |
| 38 | `South_American_traditional_folklore_38` | 7 / 1 / 1 | 649 / 74 / 106 | 352 / 44 / 44 |
| 39 | `South_European_folklore_39` | 19 / 2 / 3 | 1774 / 318 / 328 | 236 / 29 / 30 |
| 40 | `Western_songwriting_40` | 201 / 25 / 25 | 12332 / 1458 / 1608 | 140 / 17 / 18 |

## Metadata

UTF-8 CSV files, comma-separated, with Windows line ends (CRLF).

`metadata_music.csv`, one row per segment:

| Column | Content |
|---|---|
| `file` | path of the segment (see [File names](#file-names)) |
| `class` | class name |
| `split` | set: `train`, `val` or `test` |
| `info` | original file name of the track, tidied |
| `title`, `artist`, `album`, `date`, `composer` | from the tags of the original file of the track, cleaned and completed with the information of its name; in classical music `artist` can hold the performer or the composer |
| `segment` | number of the segment in its track (see [File names](#file-names)) |

Filled values: `info` 180165, `title` 179214, `artist` 124370, `album`
49140, `date` 27090, `composer` 57149 (of 181352 segments).

```
file,class,split,info,title,artist,album,date,composer,segment
Entire_music/val/Rock_35/350014_001,Rock,val,01-Hells Bells,Hells Bells,AC/DC,Back In Black,1980,,1
Entire_music/val/Baroque_sacred_06/060001_001,Baroque sacred,val,"01 - Prelude in E Flat Major, BWV 552","Prelude in E Flat Major, BWV 552",Wolfgang Ruebsam,"Clavierübung III, Vol.1 - J.S.Bach",1994,Johann Sebastian Bach,1
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
one NumPy array in each `.npz` file.

- **Images** (`ib_` files): `imagebind`, 1024 values, from the vision
  encoder of ImageBind huge (OpenCLIP ViT-H/14); the image is resized to 224
  pixels on its shorter side (bicubic), center-cropped to 224 × 224 and
  normalized as in CLIP.
- **Music**: one vector per model for every piece of 10 s that is not silent
  (mean amplitude of -60 dBFS or more), 542829 pieces.
  - `ib_` files: `imagebind`, 1024 values, from the audio encoder of
    ImageBind huge, in the same space as the image vectors; first channel at
    16 kHz, 3 clips of 2 s, each turned into a 128-bin filterbank as in
    ImageBind, mean of the 3 clip vectors;
  - `c_` files: `clap`, 512 values, audio embedding of
    `laion/larger_clap_music_and_speech` (revision `195c3a3`), first channel
    at 48 kHz.

```python
import csv
import numpy as np

def image_vector(path):
    folder, name = path.rsplit('/', 1)
    return np.load(f"{folder}/ib_{name.split('.')[0]}.npz")['imagebind']

rows = list(csv.DictReader(open('metadata_images.csv', encoding='utf-8')))
images = np.stack([image_vector(r['file']) for r in rows])
piece = np.load('Entire_music/val/Rock_35/ib_350014_001_1.npz')['imagebind']

nearest = np.argsort(images @ piece)[::-1][:10]
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
