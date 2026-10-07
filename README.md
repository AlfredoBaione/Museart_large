# Museart_Large

**A refined music-image dataset extending Museart (https://www.kaggle.com/datasets/alfredobaione/museart, https://doi.org/10.1145/3750069.3755967)**

Museart_Large contains music recordings and images of artworks, organized in the
same 41 classes (genres, historical periods and traditions of music, from
*Pre-renaissance sacred* to *Heavy metal*):

- **5414 full-length tracks**, 1543.66 hours, in their original formats
  (126.7 GiB);
- **26498 images** (3.7 GiB);
- both split into training, validation and test sets (80% / 10% / 10% of
  every class); files found to share their content (the same recording, the
  same artwork) are always in the same set;
- two CSV files with the class, the set and the available information for
  every file.

| | Where |
|---|---|
| Data | Kaggle: **[TODO: link]** |
| Preprocessing code (`preprocess_example.py`) | GitHub: https://github.com/AlfredoBaione/Museart_large |
| Paper | Museart: A Refined Music-Image Dataset (CHItaly '25), https://doi.org/10.1145/3750069.3755967; see [Citation](#citation) |

## Contents

```
Entire_music/
    train/             41 class folders, 4323 audio files
    val/               41 class folders, 538 audio files
    test/              41 class folders, 553 audio files
Entire_images/
    train/             41 class folders, 21197 images
    val/               41 class folders, 2644 images
    test/              41 class folders, 2657 images
metadata_music.csv     one row per audio file
metadata_images.csv    one row per image
```

The GitHub repository has the code, the two CSV files and an example of the
data in the same folders: for every class, one track and one image in each
of `train/`, `val/` and `test/` (123 tracks in `Entire_music/`, 123 images in
`Entire_images/`). The track is the shortest of the class in that split and
the image is the one of median file size. In 8 cases even the shortest track
is above GitHub's 100 MB file limit, so the repository has only its first
5 minutes, cut without re-encoding: `val/` Country_10, Flemish_secular_20,
Latin-American_25, North_American_traditional_folklore_28 and
Pre-renaissance_sacred_30; `test/` Classical_sacred_08, Country_10 and
Pre-renaissance_sacred_30. The examples take 2.7 GB in all. The complete
data are on Kaggle.

The files are released as they are: no format conversion, resampling,
loudness normalization or cutting. `preprocess_example.py` shows how the
audio can be prepared for training: it cuts every track into 30-second
segments and keeps them in the set of their track (see
[Preparing the audio](#preparing-the-audio-preprocess_examplepy)).

### File and folder names

A path is `<set>/<split>/<class folder>/<file>`, with `<split>` = `train`,
`val` or `test`. Each class has a two-digit code, `00` to `40` (classes in
alphabetical order), and a folder with the same name in every split of both
sets: the class name with spaces replaced by `_`, commas removed, and the code
at the end (`Rock_35`, `Rap_hip_hop_32`). Each file is named with the class
code and a four-digit number, followed by the original extension in lower
case:

```
Entire_music/val/Rock_35/350013.mp3
Entire_images/train/Baroque_sacred_06/060632.jpg
```

In every class the files are numbered from `0001` without gaps, in the
order of their original names. The numbers belong to the class, not to the
split folder, so the numbers of a class are spread over its three split
folders. Audio files and images are numbered separately:
`Entire_music/.../350013.mp3` and `Entire_images/.../350013.jpg` are not
related.

## The 41 classes

Hours: duration of the audio, measured by decoding every file with ffmpeg.

| Code | Class | Folder | Tracks | Hours | Images |
|---:|---|---|---:|---:|---:|
| 00 | 20th century experimental chamber | `20th_century_experimental_chamber_00` | 115 | 57.93 | 1574 |
| 01 | 20th century experimental orchestral | `20th_century_experimental_orchestral_01` | 118 | 56.87 | 1495 |
| 02 | 20th century traditional chamber | `20th_century_traditional_chamber_02` | 44 | 22.86 | 819 |
| 03 | 20th century traditional orchestral | `20th_century_traditional_orchestral_03` | 55 | 38.64 | 885 |
| 04 | African traditional folklore | `African_traditional_folklore_04` | 9 | 7.93 | 1195 |
| 05 | Afro-American | `Afro-American_05` | 90 | 20.51 | 695 |
| 06 | Baroque sacred | `Baroque_sacred_06` | 96 | 23.06 | 701 |
| 07 | Baroque secular | `Baroque_secular_07` | 1209 | 84.69 | 868 |
| 08 | Classical sacred | `Classical_sacred_08` | 5 | 6.37 | 553 |
| 09 | Classical secular | `Classical_secular_09` | 58 | 23.74 | 405 |
| 10 | Country | `Country_10` | 11 | 14.19 | 111 |
| 11 | Dance | `Dance_11` | 345 | 110.23 | 214 |
| 12 | Early 20th century experimental chamber | `Early_20th_century_experimental_chamber_12` | 8 | 5.22 | 490 |
| 13 | Early 20th century experimental orchestral | `Early_20th_century_experimental_orchestral_13` | 37 | 18.68 | 476 |
| 14 | Early 20th century traditional chamber | `Early_20th_century_traditional_chamber_14` | 75 | 31.61 | 874 |
| 15 | Early 20th century traditional orchestral | `Early_20th_century_traditional_orchestral_15` | 31 | 32.00 | 922 |
| 16 | East Asian traditional folklore | `East_Asian_traditional_folklore_16` | 18 | 18.76 | 844 |
| 17 | East European folklore | `East_European_folklore_17` | 24 | 18.15 | 534 |
| 18 | Electronic | `Electronic_18` | 77 | 38.20 | 1600 |
| 19 | Flemish sacred | `Flemish_sacred_19` | 5 | 3.98 | 362 |
| 20 | Flemish secular | `Flemish_secular_20` | 9 | 6.19 | 370 |
| 21 | Heavy metal | `Heavy_metal_21` | 105 | 64.17 | 1330 |
| 22 | Jazz | `Jazz_22` | 1443 | 212.71 | 303 |
| 23 | Late romanticism chamber | `Late_romanticism_chamber_23` | 19 | 13.93 | 699 |
| 24 | Late romanticism orchestral | `Late_romanticism_orchestral_24` | 17 | 19.81 | 1133 |
| 25 | Latin-American | `Latin-American_25` | 25 | 24.50 | 857 |
| 26 | Middle, north European folklore | `Middle_north_European_folklore_26` | 45 | 11.62 | 431 |
| 27 | Middle, south Asian traditional folklore | `Middle_south_Asian_traditional_folklore_27` | 23 | 23.26 | 490 |
| 28 | North American traditional folklore | `North_American_traditional_folklore_28` | 3 | 4.08 | 433 |
| 29 | Pop | `Pop_29` | 44 | 55.41 | 509 |
| 30 | Pre-renaissance sacred | `Pre-renaissance_sacred_30` | 4 | 5.04 | 399 |
| 31 | Pre-renaissance secular | `Pre-renaissance_secular_31` | 7 | 7.66 | 187 |
| 32 | Rap, hip hop | `Rap_hip_hop_32` | 361 | 82.81 | 132 |
| 33 | Renaissance sacred | `Renaissance_sacred_33` | 6 | 6.19 | 508 |
| 34 | Renaissance secular | `Renaissance_secular_34` | 25 | 7.99 | 685 |
| 35 | Rock | `Rock_35` | 473 | 160.00 | 258 |
| 36 | Romanticism chamber | `Romanticism_chamber_36` | 75 | 29.04 | 687 |
| 37 | Romanticism orchestral | `Romanticism_orchestral_37` | 15 | 13.13 | 459 |
| 38 | South American traditional folklore | `South_American_traditional_folklore_38` | 14 | 11.36 | 541 |
| 39 | South European folklore | `South_European_folklore_39` | 28 | 21.17 | 295 |
| 40 | Western songwriting | `Western_songwriting_40` | 243 | 129.97 | 175 |
| | **Total** | | **5414** | **1543.66** | **26498** |

## Training, validation and test sets

- **Per track and per image.** Each audio file and each image is in one set
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

| Set | Tracks | Hours | Images |
|---|---:|---:|---:|
| training (`train`) | 4323 (79.8%) | 1224.17 (79.3%) | 21197 (80.0%) |
| validation (`val`) | 538 (9.9%) | 144.50 (9.4%) | 2644 (10.0%) |
| test (`test`) | 553 (10.2%) | 174.99 (11.3%) | 2657 (10.0%) |
| total | 5414 | 1543.66 | 26498 |

The proportions are counted on files, so the hours are close to, but not
exactly, 80% / 10% / 10%.

Per class (training / validation / test):

| Code | Folder | Tracks | Hours | Images |
|---:|---|---:|---:|---:|
| 00 | `20th_century_experimental_chamber_00` | 92 / 11 / 12 | 44.28 / 4.09 / 9.57 | 1259 / 157 / 158 |
| 01 | `20th_century_experimental_orchestral_01` | 94 / 12 / 12 | 46.60 / 6.35 / 3.92 | 1196 / 149 / 150 |
| 02 | `20th_century_traditional_chamber_02` | 35 / 4 / 5 | 18.04 / 2.84 / 1.98 | 655 / 82 / 82 |
| 03 | `20th_century_traditional_orchestral_03` | 44 / 5 / 6 | 32.71 / 3.21 / 2.72 | 708 / 88 / 89 |
| 04 | `African_traditional_folklore_04` | 7 / 1 / 1 | 7.08 / 0.63 / 0.22 | 956 / 119 / 120 |
| 05 | `Afro-American_05` | 72 / 9 / 9 | 14.86 / 0.77 / 4.89 | 556 / 69 / 70 |
| 06 | `Baroque_sacred_06` | 77 / 9 / 10 | 17.79 / 0.71 / 4.56 | 561 / 70 / 70 |
| 07 | `Baroque_secular_07` | 967 / 121 / 121 | 67.23 / 7.94 / 9.52 | 694 / 87 / 87 |
| 08 | `Classical_sacred_08` | 3 / 1 / 1 | 3.75 / 0.92 / 1.70 | 443 / 55 / 55 |
| 09 | `Classical_secular_09` | 46 / 6 / 6 | 19.16 / 0.91 / 3.67 | 324 / 40 / 41 |
| 10 | `Country_10` | 9 / 1 / 1 | 11.10 / 1.50 / 1.59 | 89 / 11 / 11 |
| 11 | `Dance_11` | 276 / 34 / 35 | 79.72 / 19.57 / 10.95 | 171 / 21 / 22 |
| 12 | `Early_20th_century_experimental_chamber_12` | 6 / 1 / 1 | 3.55 / 1.21 / 0.45 | 392 / 49 / 49 |
| 13 | `Early_20th_century_experimental_orchestral_13` | 29 / 4 / 4 | 15.95 / 1.44 / 1.30 | 381 / 47 / 48 |
| 14 | `Early_20th_century_traditional_chamber_14` | 60 / 7 / 8 | 29.11 / 0.80 / 1.70 | 699 / 87 / 88 |
| 15 | `Early_20th_century_traditional_orchestral_15` | 25 / 3 / 3 | 28.69 / 0.93 / 2.39 | 738 / 92 / 92 |
| 16 | `East_Asian_traditional_folklore_16` | 14 / 2 / 2 | 13.81 / 2.41 / 2.54 | 675 / 84 / 85 |
| 17 | `East_European_folklore_17` | 19 / 2 / 3 | 13.74 / 1.53 / 2.87 | 427 / 53 / 54 |
| 18 | `Electronic_18` | 61 / 8 / 8 | 28.72 / 6.32 / 3.16 | 1280 / 160 / 160 |
| 19 | `Flemish_sacred_19` | 3 / 1 / 1 | 2.62 / 0.52 / 0.84 | 290 / 36 / 36 |
| 20 | `Flemish_secular_20` | 7 / 1 / 1 | 4.88 / 1.23 / 0.08 | 296 / 37 / 37 |
| 21 | `Heavy_metal_21` | 84 / 10 / 11 | 53.09 / 4.54 / 6.53 | 1064 / 133 / 133 |
| 22 | `Jazz_22` | 1155 / 144 / 144 | 173.34 / 17.53 / 21.84 | 243 / 30 / 30 |
| 23 | `Late_romanticism_chamber_23` | 15 / 2 / 2 | 10.65 / 0.84 / 2.44 | 559 / 70 / 70 |
| 24 | `Late_romanticism_orchestral_24` | 13 / 2 / 2 | 17.07 / 1.77 / 0.96 | 907 / 113 / 113 |
| 25 | `Latin-American_25` | 20 / 2 / 3 | 19.55 / 2.48 / 2.47 | 685 / 86 / 86 |
| 26 | `Middle_north_European_folklore_26` | 36 / 4 / 5 | 7.60 / 1.51 / 2.51 | 345 / 43 / 43 |
| 27 | `Middle_south_Asian_traditional_folklore_27` | 19 / 2 / 2 | 19.04 / 1.82 / 2.40 | 392 / 49 / 49 |
| 28 | `North_American_traditional_folklore_28` | 1 / 1 / 1 | 0.91 / 2.18 / 0.99 | 347 / 43 / 43 |
| 29 | `Pop_29` | 35 / 4 / 5 | 49.52 / 1.05 / 4.83 | 406 / 52 / 51 |
| 30 | `Pre-renaissance_sacred_30` | 2 / 1 / 1 | 2.63 / 1.17 / 1.23 | 319 / 40 / 40 |
| 31 | `Pre-renaissance_secular_31` | 5 / 1 / 1 | 5.85 / 0.71 / 1.10 | 149 / 19 / 19 |
| 32 | `Rap_hip_hop_32` | 289 / 36 / 36 | 68.56 / 8.05 / 6.20 | 106 / 13 / 13 |
| 33 | `Renaissance_sacred_33` | 4 / 1 / 1 | 4.38 / 0.90 / 0.91 | 406 / 51 / 51 |
| 34 | `Renaissance_secular_34` | 20 / 2 / 3 | 7.89 / 0.04 / 0.06 | 548 / 68 / 69 |
| 35 | `Rock_35` | 379 / 47 / 47 | 119.14 / 15.36 / 25.50 | 206 / 26 / 26 |
| 36 | `Romanticism_chamber_36` | 60 / 7 / 8 | 22.50 / 0.37 / 6.18 | 549 / 69 / 69 |
| 37 | `Romanticism_orchestral_37` | 12 / 1 / 2 | 9.66 / 1.12 / 2.35 | 367 / 46 / 46 |
| 38 | `South_American_traditional_folklore_38` | 11 / 1 / 2 | 9.40 / 0.62 / 1.34 | 433 / 54 / 54 |
| 39 | `South_European_folklore_39` | 22 / 3 / 3 | 16.29 / 3.23 / 1.65 | 236 / 29 / 30 |
| 40 | `Western_songwriting_40` | 195 / 24 / 24 | 103.71 / 13.38 / 12.87 | 140 / 17 / 18 |

## metadata_music.csv

One row per audio file. UTF-8, comma-separated (values containing a comma are
in double quotes), Windows line ends (CRLF).

| Column | Content |
|---|---|
| `file` | path of the audio file, from the dataset root |
| `class` | class name |
| `split` | `train`, `val` or `test`: the set of the file (the same as its folder) |
| `info` | the original file name without extension. Names that were hard to read were tidied (emoji, decorative letters, underscores used as spaces), signatures and web addresses were removed, unreadable names were rebuilt from the tags, and names with no information were left empty |
| `title`, `artist`, `album`, `date`, `composer` | from the tags of the original file. Text that is not about the music was removed (signatures, web addresses, values written by software such as `Unknown Artist` or `Track 5`, the name of the uploader and the upload date), and broken accents were repaired. Empty fields were then filled, where possible, with what the original file name says |

Values are written as they were found: in classical music `artist` can hold
the performer or the composer; `date` is a year for 2057 tracks and a full
date or other text (such as `90s`) for 12.

Filled values, out of 5414 tracks:

| `info` | `title` | `artist` | `album` | `date` | `composer` |
|---:|---:|---:|---:|---:|---:|
| 5242 | 5171 | 4268 | 3406 | 2069 | 1571 |

172 tracks have none of the five tag fields; 172 have an empty `info`.

```
file,class,split,info,title,artist,album,date,composer
Entire_music/val/Rock_35/350013.mp3,Rock,val,01-Hells Bells,Hells Bells,AC/DC,Back In Black,1980,
Entire_music/val/Baroque_sacred_06/060001.mp3,Baroque sacred,val,"01 - Prelude in E Flat Major, BWV 552","Prelude in E Flat Major, BWV 552",Wolfgang Ruebsam,"Clavierübung III, Vol.1 - J.S.Bach",1994,Johann Sebastian Bach
Entire_music/train/Baroque_sacred_06/060003.mp3,Baroque sacred,train,,,,,,
```

## metadata_images.csv

One row per image, same format as above.

| Column | Content |
|---|---|
| `file` | path of the image, from the dataset root |
| `class` | class name |
| `split` | `train`, `val` or `test`: the set of the file (the same as its folder) |
| `info` | artist, title and year read from the original file name, written as `Artist - Title (year)` or with the parts that are known; empty when the name carried no information (1660 images whose names were codes or hashes) |

```
file,class,split,info
Entire_images/train/Baroque_sacred_06/060632.jpg,Baroque sacred,train,Jacob Jordaens - Bust of Satyr (1621)
Entire_images/val/Electronic_18/180321.jpg,Electronic,val,Andy Warhol
Entire_images/train/Heavy_metal_21/210001.jpg,Heavy metal,train,
```

## Audio files

| mp3 | wma | mpc | m4a | ape | ogg | flac | wav | aiff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5068 | 115 | 61 | 59 | 54 | 16 | 16 | 14 | 11 |

The audio data are those of the original files. The tags were rewritten
without re-encoding: inside each file they hold the same five values as the
CSV (`title`, `artist`, `album`, `date`, `composer`, where the format allows
it), and nothing else apart from technical fields used by decoders (such as
the encoder delay of m4a files). Covers, comments, genres, track numbers and
all other tags were removed.

Known problems, measured with ffmpeg:

- 276 files contain damaged frames, which decoders skip (56.31 hours in
  these files). One of them, `Entire_music/test/Jazz_22/220759.mp3`, can be
  decoded only for its first 76.6 s, while its header states 304.0 s.
- `Entire_music/val/Romanticism_chamber_36/360017.m4a` and
  `Entire_music/train/Romanticism_chamber_36/360020.m4a` contain damaged AAC
  data: ffmpeg decodes them with many errors and ends with an error code;
  `preprocess_example.py` skips them.
- In `Entire_music/train/Pre-renaissance_sacred_30/300001.mp3` (1 h 43 min)
  the right channel is the left one with the opposite sign: each channel
  alone is normal, but their average (a mono mix) is silence.
  `preprocess_example.py` keeps the two channels apart.
- The header of some mp3 files states a wrong duration (one states 1058.5 s
  and contains 643.6 s). The hours in this README were measured by decoding.

## Images

| jpg | jpeg | png |
|---:|---:|---:|
| 26443 | 31 | 24 |

The image files are the original ones, byte for byte.

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
- Across classes, 1548 images appear as identical files in two or more
  classes (1569 extra copies), and one track appears in two classes
  (`Entire_music/train/Rap_hip_hop_32/320231.mp3` and
  `Entire_music/train/Rock_35/350275.mp3`). Each copy is a file of its own
  class, with its own number.
- 594 pairs of images inside a class, and 245 pairs across classes, are
  near-duplicates (perceptual hashes: pHash distance up to 8 and dHash
  distance up to 12), very probably the same artwork in two different
  reproductions. They were kept.
- 182 pairs of tracks of different length share part of their audio: a track
  and a longer file that contains it (a whole album in one file), an edit and
  its full version. 5 pairs of tracks of the same length share only part of
  their first 2 minutes (from 10 s to 79% of them). They were kept.

All the files with the same content are in the same set (training,
validation or test): 65 groups of tracks (251 tracks) and 1947 groups of
images (4090 images).

What these checks cannot see: two reproductions of the same artwork that
differ a lot (a strong crop, a photograph taken at an angle), and two tracks
that share audio only after their first 2 minutes. Different performances of
the same piece are different recordings and are not grouped.

## Preparing the audio: preprocess_example.py

`preprocess_example.py` (GitHub: https://github.com/AlfredoBaione/Museart_large) is an example of how the
audio can be prepared for training: it turns `Entire_music` into mono
segments of 30 seconds (`CHUNK_LENGTH_SEC` at the top of the script), each in
the set of its track. Its acoustic rules are those of the authors'
audio-generation pipeline.

### Requirements

- Python 3.9 or later, with numpy, soundfile and tqdm:
  `pip install numpy soundfile tqdm`;
- `ffmpeg` and `ffprobe` on the PATH.

Tested on Windows 11 with Python 3.14, numpy 2.4, soundfile 0.13, tqdm 4.67
and an ffmpeg build of March 2026.

### Usage

```
python preprocess_example.py --source path/to/Entire_music --output path/to/Entire_music_30sec_splits --workers 8
```

| Option | Meaning |
|---|---|
| `--source` | the `Entire_music` folder, with `train/`, `val/` and `test/` inside (default: `Entire_music`) |
| `--output` | output folder (default: `Entire_music_30sec_splits`) |
| `--workers` | tracks processed in parallel (default: number of CPU cores minus one, at least 2) |
| `--format` | `flac` (default) or `wav`: the same samples, FLAC takes about half the space |
| `--dry_run` | only count the tracks per set and class; nothing is decoded |
| `--force` | process again the tracks already done |

The work is done by ffmpeg on the CPU (the GPU is not used). An interrupted
run continues where it stopped when the same command is run again
(`manifest.jsonl` records the tracks already done).

### What it does to each track

1. One pass with ffmpeg finds the leading and trailing silence (below -55 dB
   for at least 0.1 s) and measures the integrated loudness and the true peak
   of the track (EBU R128, `loudnorm` used only to measure).
2. **One constant gain** for the whole track: gain (dB) = min(-18 - measured
   loudness, -1 - measured true peak). The track reaches -18 LUFS unless its
   true peak would go above -1 dBTP; it is never compressed. If the
   measurement fails, no gain is applied.
3. **Stereo: each channel becomes its own mono example** (channels 0 and 1),
   instead of the average of the two, which cancels out when the channels are
   in opposite phase (see [Audio files](#audio-files)). Mono files give one
   example.
4. For every channel, one more ffmpeg pass removes the silence, applies the
   gain and converts the audio to 44 100 Hz, 16 bit; the channel is cut into
   consecutive segments of exactly 30 s (1 323 000 samples). The last part,
   shorter than 30 s, is discarded, and so is every segment whose mean
   amplitude is below -60 dBFS (silence).

The values are at the top of the script (`SILENCE_TRIM_DB`, `TARGET_LUFS`,
`TARGET_TP`, `SILENCE_THRESH_DB`, `STEREO_SPLIT`). Tracks shorter than 30 s
(also after the silence is removed) and files that ffmpeg cannot decode are
listed in `skipped.csv` with the reason.

5406 of the 5414 tracks are stereo, so the output has fewer than 371 000
segments (1543.66 hours × 2 channels / 30 s); a 30-s WAV segment takes
2.6 MB, so the WAV output needs up to about 1 TB, and the FLAC output about
half of that.

### Output

| Path | Content |
|---|---|
| `<split>/<class folder>/<number>_<hash>_ch<C>_<NNNN>.flac` | the segments, in the set of their track; `<number>` is the track (`350013`), `<hash>` 6 characters from its path, `<C>` the channel (`0` or `1`; `0` for mono files), `NNNN` the position of the segment in the track (`0000` = first 30 s after the leading silence) |
| `metadata.csv` | one row per segment: `filename`, `split`, `class_name`, `source_name`, `source_path` (the track, e.g. `val/Rock_35/350013.mp3`), `channel`, `seg_index`, `start_sec`, `duration_sec`, `gain_db` (the gain of the track), `mean_dbfs` (mean amplitude of the segment) |
| `skipped.csv` | the tracks not used, with the reason |
| `manifest.jsonl` | progress of the run |

The class of a segment (folders and `class_name`) is the folder name
(`Rock_35`); `metadata_music.csv` gives the class name of every folder. The
script does not split anything: all the segments of a track, of both its
channels, are in the set of the track, so no track is in two sets.

## License

- Code (`preprocess_example.py`): MIT License, see `LICENSE` in the GitHub
  repository.
- Data: MIT License, see `LICENSE` in the GitHub
  repository.

## Citation

**Alfredo Baione and Genoveffa Tortora. 2025. Museart: A Refined Music-Image Dataset. In Proceedings of the 16th Biannual Conference of the Italian SIGCHI Chapter (CHItaly '25). Association for Computing Machinery, New York, NY, USA, Article 107, 1–3. https://doi.org/10.1145/3750069.3755967**
