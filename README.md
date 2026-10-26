# Museart_Large

**A refined music-image dataset extending Museart (https://www.kaggle.com/datasets/alfredobaione/museart, https://doi.org/10.1145/3750069.3755967)**

Museart_Large describes music recordings and images of artworks, organized in
the same 41 classes (genres, historical periods and traditions of music, from
*Pre-renaissance sacred* to *Heavy metal*):

- **5413 full-length tracks** (1543.60 hours), released as metadata and as
  vectors of 49481 pieces of 10 seconds, computed with ImageBind and CLAP;
- **25683 images**, released as metadata and ImageBind vectors; **14546** of
  them, reproductions of works whose authors died in 1955 or earlier, are
  also released as image files (2.3 GiB);
- both split into training, validation and test sets (80% / 10% / 10% of
  every class); files found to share their content (the same recording, the
  same artwork) are always in the same set;
- two CSV files with the class, the set and the available information for
  every track and image.

Audio files and images of works still protected by copyright are not
distributed (see [Copyright](#copyright)).

| | Where |
|---|---|
| Data | Kaggle: **https://www.kaggle.com/datasets/alfredobaione/museart-large** |
| Code (technical validation) | GitHub: https://github.com/AlfredoBaione/Museart_large |
| Paper | Museart: A Refined Music-Image Dataset (CHItaly '25), https://doi.org/10.1145/3750069.3755967; see [Citation](#citation) |

## Contents

```
Entire_images/
    train/                        41 class folders, 20546 files of vectors, 11650 images
    val/                          41 class folders, 2563 files of vectors, 1451 images
    test/                         41 class folders, 2574 files of vectors, 1445 images
Entire_music/
    train/                        41 class folders, 4322 files of vectors
    val/                          41 class folders, 542 files of vectors
    test/                         41 class folders, 549 files of vectors
metadata_images.csv               one row per image (25683)
metadata_music.csv                one row per track (5413)
```

Nine classes have no image files (see [The 41 classes](#the-41-classes)). The
image files are the original ones, byte for byte.

The GitHub repository has the code, the two CSV files and an example of the
image files: one image for each class and split that has image files (94
images in `Entire_images/`). The complete data are on Kaggle.

### File and folder names

Every track and image has a path, given in the `file` column of the CSV
files: `<set>/<split>/<class folder>/<file>`, with `<set>` = `Entire_music` or
`Entire_images` and `<split>` = `train`, `val` or `test`. The vectors of every
track and image are in the file with the same path and the extension `.npz`.
The audio files are not distributed, and the image files only for the images
released as files. Each class has a two-digit code, `00` to `40` (classes in
alphabetical order), and a folder name: the class name with spaces replaced
by `_`, commas removed, and the code at the end (`Rock_35`,
`Rap_hip_hop_32`). Each file is named with the class code and a four-digit
number, followed by the original extension in lower case:

```
Entire_music/val/Rock_35/350014.mp3                a track
Entire_music/val/Rock_35/350014.npz                its vectors
Entire_images/train/Baroque_sacred_06/060769.jpg   an image
Entire_images/train/Baroque_sacred_06/060769.npz   its vectors
```

In every class the files are numbered from `0001` without gaps, in the
order of their original names. The numbers belong to the class, not to the
split folder, so the numbers of a class are spread over its three split
folders. Tracks and images are numbered separately:
`Entire_music/.../350014.mp3` and `Entire_images/.../350014.jpg` are not
related.

## The 41 classes

In the historical classes, a track belongs to the period in which its work
was composed, and an image to the period in which its artist mainly worked.
Periods: Pre-renaissance up to 1419; Renaissance 1420–1599 (Flemish:
composers and painters of the Low Countries); Baroque 1600–1749; Classical
1750–1819 (painting: up to 1799); Romanticism 1820–1849 (painting: from
1800); Late romanticism 1850–1899; Early 20th century 1900–1929; 20th century
from 1930.

Hours: duration of the audio, measured by decoding every file with ffmpeg.
Image files: images also released as files.

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

- **Per track and per image.** Each track and each image is in one set
  only.
- **80% / 10% / 10% of every class** (stratified by class, rounded with the
  largest remainder); every class with at least 3 files has at least one in
  validation and one in test.
- **Files with the same content in the same set.** Identical images (also
  when they are in different classes), images that look like the same
  artwork in another file (perceptual hashes), and tracks that share their
  audio (the same recording in two classes, a track contained in a longer
  file such as a whole album, an edit and its full version; found with an
  acoustic fingerprint) were kept together. See [Duplicates](#duplicates) for
  how they were found and what these checks cannot see.
- Random order with a fixed seed (42).

| Set | Tracks | Hours | Images | Image files |
|---|---:|---:|---:|---:|
| training (`train`) | 4322 (79.8%) | 1217.66 (78.9%) | 20546 (80.0%) | 11650 |
| validation (`val`) | 542 (10.0%) | 151.99 (9.8%) | 2563 (10.0%) | 1451 |
| test (`test`) | 549 (10.1%) | 173.95 (11.3%) | 2574 (10.0%) | 1445 |
| total | 5413 | 1543.60 | 25683 | 14546 |

The proportions are counted on files, so the hours are close to, but not
exactly, 80% / 10% / 10%.

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

## metadata_music.csv

One row per track. UTF-8, comma-separated (values containing a comma are in
double quotes), Windows line ends (CRLF).

| Column | Content |
|---|---|
| `file` | path of the track (see [File and folder names](#file-and-folder-names)); its vectors are at the same path with the extension `.npz` |
| `class` | class name |
| `split` | `train`, `val` or `test`: the set of the file (the same as its folder) |
| `info` | the original file name without extension. Names that were hard to read were tidied (emoji, decorative letters, underscores used as spaces), signatures and web addresses were removed, unreadable names were rebuilt from the tags, and names with no information were left empty |
| `title`, `artist`, `album`, `date`, `composer` | from the tags of the original file. Text that is not about the music was removed (signatures, web addresses, values written by software such as `Unknown Artist` or `Track 5`, the name of the uploader and the upload date), and broken accents were repaired. Empty fields were then filled, where possible, with what the original file name says |

Values are written as they were found: in classical music `artist` can hold
the performer or the composer; `date` is a year for 2055 tracks and a full
date or other text (such as `90s`) for 12.

Filled values, out of 5413 tracks:

| `info` | `title` | `artist` | `album` | `date` | `composer` |
|---:|---:|---:|---:|---:|---:|
| 5241 | 5170 | 4267 | 3404 | 2067 | 1569 |

172 tracks have none of the five tag fields; 172 have an empty `info`.

```
file,class,split,info,title,artist,album,date,composer
Entire_music/val/Rock_35/350014.mp3,Rock,val,01-Hells Bells,Hells Bells,AC/DC,Back In Black,1980,
Entire_music/val/Baroque_sacred_06/060001.mp3,Baroque sacred,val,"01 - Prelude in E Flat Major, BWV 552","Prelude in E Flat Major, BWV 552",Wolfgang Ruebsam,"Clavierübung III, Vol.1 - J.S.Bach",1994,Johann Sebastian Bach
Entire_music/train/Baroque_sacred_06/060003.mp3,Baroque sacred,train,,,,,,
```

## metadata_images.csv

One row per image, same format as above.

| Column | Content |
|---|---|
| `file` | path of the image; its vectors are at the same path with the extension `.npz`, and the image file is there only for the images released as files |
| `class` | class name |
| `split` | `train`, `val` or `test`: the set of the file (the same as its folder) |
| `info` | artist, title and year read from the original file name, written as `Artist - Title (year)` or with the parts that are known; empty when the name carried no information (1673 images whose names carried no information, such as codes or hashes) |

```
file,class,split,info
Entire_images/train/Baroque_sacred_06/060769.jpg,Baroque sacred,train,Jacob Jordaens - Bust of Satyr (1621)
Entire_images/val/Pop_29/290001.jpg,Pop,val,Andy Warhol
Entire_images/train/Heavy_metal_21/210001.jpg,Heavy metal,train,
```

## Vectors

The vectors are those of the technical validation (code in
`technical_validation/` of the GitHub repository): L2-normalized, float32,
stored as NumPy arrays in `.npz` files.

- **Images** (`Entire_images/`): one `.npz` file per image, with the array
  `imagebind` (1024 values). ImageBind (huge), vision encoder (OpenCLIP
  ViT-H/14); each image was resized to 224 pixels on its shorter side
  (bicubic), center-cropped to 224 × 224 and normalized as in CLIP.
- **Music** (`Entire_music/`): one `.npz` file per track, with three arrays
  of one row per piece: `imagebind` (pieces × 1024), `clap` (pieces × 512)
  and `start_s`, the start of the piece in seconds. From every track, pieces
  of 10 s were taken, one for every 2 minutes of music and at least 3 per
  track, centered in equal parts of the track. Pieces with a mean amplitude
  below -60 dBFS and pieces that could not be decoded were discarded: 49481
  of 49844 pieces remain, from all the 5413 tracks.
- **ImageBind** (`imagebind`, same space as the image vectors): the first
  channel of the piece at 16 kHz, cut into 3 clips of 2 s at uniform start
  times, each turned into a 128-bin filterbank as in ImageBind; the vector of
  the piece is the mean of the 3 clip embeddings.
- **CLAP** (`clap`): audio embedding of `laion/larger_clap_music_and_speech`
  (revision `195c3a3`) for the first channel of the piece at 48 kHz.

```python
import csv
import numpy as np

rows = list(csv.DictReader(open('metadata_images.csv', encoding='utf-8')))
images = np.stack([np.load(r['file'].rsplit('.', 1)[0] + '.npz')['imagebind'] for r in rows])
track = np.load('Entire_music/val/Rock_35/350014.npz')

vector = track['imagebind'].mean(axis=0)
nearest = np.argsort(images @ vector)[::-1][:10]
print([rows[i]['info'] for i in nearest])
```

## Copyright

- **Images.** An image is released as a file when the author of the work
  died in 1955 or earlier. In the European Union the rights of an author run
  for the life of the author and 70 years after the death, counted from
  1 January of the following year (Directive 2006/116/EC, art. 1 and 8), and
  faithful reproductions of works of visual art whose protection has expired
  are not protected (Directive (EU) 2019/790, art. 14). The date of death of
  the author named in `info` was taken from Wikidata and checked by hand when
  the name was ambiguous or misspelled. Only the vector is released for
  authors who died in 1956 or later or may be alive, for images whose author
  is not identified (except in the classes of periods before 1850,
  Pre-renaissance to Romanticism), for images credited to several authors not
  all identified, and for photographs of three-dimensional works
  (sculptures, objects, buildings), whose photographer may hold rights.
- **Music.** No audio is released: the recordings, and many of the
  compositions, are protected. Every track is described by its metadata and
  its vectors.

## Images

| jpg | jpeg | png | webp |
|---:|---:|---:|---:|
| 14516 | 8 | 17 | 5 |

## Duplicates

- Inside a class there are no identical files (images: identical bytes;
  audio: identical audio data).
- Inside a class there are no two files with the same recording, also when
  the two files are encoded differently (another format or bitrate, another
  copy of the same CD, the same song on two compilations). These copies were
  found by comparing an acoustic fingerprint of the tracks (Haitsma & Kalker,
  2002): for every track, its first 2 minutes were searched in all the other
  tracks, in blocks of 5 s. 574 copies (34.25 hours) were removed; of every
  recording the copy with the best audio quality was kept.
- Across classes, 548 images appear as identical files in two or more
  classes (550 extra copies), and one track appears in two classes
  (`Entire_music/train/Rap_hip_hop_32/320231.mp3` and
  `Entire_music/train/Rock_35/350287.mp3`). Each copy is a file of its own
  class, with its own number.
- 576 pairs of images inside a class, and 196 pairs across classes, are
  near-duplicates (perceptual hashes: pHash distance up to 8 and dHash
  distance up to 12), very probably the same artwork in two different
  reproductions. They were kept.
- 182 pairs of tracks of different length share part of their audio: a track
  and a longer file that contains it (a whole album in one file), an edit and
  its full version. 5 pairs of tracks of the same length share only part of
  their first 2 minutes (from 10 s to 79% of them). They were kept.

All the files with the same content are in the same set (training,
validation or test): 65 groups of tracks (251 tracks) and 989 groups of
images (2118 images).

What these checks cannot see: two reproductions of the same artwork that
differ a lot (a strong crop, a photograph taken at an angle), and two tracks
that share audio only after their first 2 minutes. Different performances of
the same piece are different recordings and are not grouped.

## License

- Code: MIT License, see `LICENSE` in the GitHub repository.
- Data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

**Alfredo Baione and Genoveffa Tortora. 2025. Museart: A Refined Music-Image Dataset. In Proceedings of the 16th Biannual Conference of the Italian SIGCHI Chapter (CHItaly '25). Association for Computing Machinery, New York, NY, USA, Article 107, 1–3. https://doi.org/10.1145/3750069.3755967**
