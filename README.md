# Museart_Large

**A refined music-image dataset extending Museart (https://www.kaggle.com/datasets/alfredobaione/museart, https://doi.org/10.1145/3750069.3755967)**

Museart_Large contains music recordings and images of artworks, organized in the
same 41 classes (genres, historical periods and traditions of music, from
*Pre-renaissance sacred* to *Heavy metal*):

- **5414 full-length tracks**, 1543.66 hours, in their original formats
  (126.7 GiB);
- **26538 images** (3.7 GiB);
- both split into training, validation and test sets (80% / 10% / 10% of
  every class); files found to share their content (the same recording, the
  same artwork) are always in the same set;
- two CSV files with the class, the set and the available information for
  every file.

| | Where |
|---|---|
| Data | Kaggle: **https://www.kaggle.com/datasets/alfredobaione/museart-large** |
| Preprocessing code (`preprocess_example.py`) | GitHub: https://github.com/AlfredoBaione/Museart_large |
| Paper | Museart: A Refined Music-Image Dataset (CHItaly '25), https://doi.org/10.1145/3750069.3755967; see [Citation](#citation) |

## Contents

```
Entire_music/
    train/             41 class folders, 4323 audio files
    val/               41 class folders, 542 audio files
    test/              41 class folders, 549 audio files
Entire_images/
    train/             41 class folders, 21229 images
    val/               41 class folders, 2650 images
    test/              41 class folders, 2659 images
metadata_music.csv     one row per audio file
metadata_images.csv    one row per image
```

The GitHub repository has the code, the two CSV files and an example of the
data in the same folders: for every class, one track and one image in each
of `train/`, `val/` and `test/` (123 tracks in `Entire_music/`, 123 images in
`Entire_images/`). The track is the shortest of the class in that split and
the image is the one of median file size. In 10 cases even the shortest track
is above GitHub's 100 MB file limit, so the repository has only its first
5 minutes, cut without re-encoding: `train/` Classical_sacred_08; `val/`
Country_10, Flemish_secular_20, North_American_traditional_folklore_28,
Pre-renaissance_sacred_30 and Renaissance_secular_34; `test/`
Classical_sacred_08, Country_10, Pre-renaissance_sacred_30 and
Renaissance_sacred_33. The examples take 2.7 GB in all. The complete data
are on Kaggle.

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
Entire_music/val/Rock_35/350014.mp3
Entire_images/train/Baroque_sacred_06/060769.jpg
```

In every class the files are numbered from `0001` without gaps, in the
order of their original names. The numbers belong to the class, not to the
split folder, so the numbers of a class are spread over its three split
folders. Audio files and images are numbered separately:
`Entire_music/.../350014.mp3` and `Entire_images/.../350014.jpg` are not
related.

## The 41 classes

Hours: duration of the audio, measured by decoding every file with ffmpeg.

| Code | Class | Folder | Tracks | Hours | Images |
|---:|---|---|---:|---:|---:|
| 00 | 20th century experimental chamber | `20th_century_experimental_chamber_00` | 112 | 50.18 | 1146 |
| 01 | 20th century experimental orchestral | `20th_century_experimental_orchestral_01` | 114 | 59.31 | 1038 |
| 02 | 20th century traditional chamber | `20th_century_traditional_chamber_02` | 27 | 10.19 | 451 |
| 03 | 20th century traditional orchestral | `20th_century_traditional_orchestral_03` | 38 | 23.86 | 397 |
| 04 | African traditional folklore | `African_traditional_folklore_04` | 18 | 13.83 | 1102 |
| 05 | Afro-American | `Afro-American_05` | 81 | 14.51 | 743 |
| 06 | Baroque sacred | `Baroque_sacred_06` | 97 | 23.64 | 843 |
| 07 | Baroque secular | `Baroque_secular_07` | 1238 | 88.52 | 939 |
| 08 | Classical sacred | `Classical_sacred_08` | 3 | 4.34 | 370 |
| 09 | Classical secular | `Classical_secular_09` | 51 | 22.98 | 335 |
| 10 | Country | `Country_10` | 11 | 14.19 | 111 |
| 11 | Dance | `Dance_11` | 345 | 110.23 | 214 |
| 12 | Early 20th century experimental chamber | `Early_20th_century_experimental_chamber_12` | 25 | 16.18 | 681 |
| 13 | Early 20th century experimental orchestral | `Early_20th_century_experimental_orchestral_13` | 21 | 10.07 | 748 |
| 14 | Early 20th century traditional chamber | `Early_20th_century_traditional_chamber_14` | 40 | 23.87 | 356 |
| 15 | Early 20th century traditional orchestral | `Early_20th_century_traditional_orchestral_15` | 35 | 27.97 | 721 |
| 16 | East Asian traditional folklore | `East_Asian_traditional_folklore_16` | 18 | 18.76 | 870 |
| 17 | East European folklore | `East_European_folklore_17` | 25 | 19.13 | 534 |
| 18 | Electronic | `Electronic_18` | 78 | 38.28 | 1600 |
| 19 | Flemish sacred | `Flemish_sacred_19` | 5 | 3.98 | 205 |
| 20 | Flemish secular | `Flemish_secular_20` | 9 | 6.19 | 212 |
| 21 | Heavy metal | `Heavy_metal_21` | 105 | 64.17 | 1330 |
| 22 | Jazz | `Jazz_22` | 1389 | 207.02 | 303 |
| 23 | Late romanticism chamber | `Late_romanticism_chamber_23` | 83 | 38.29 | 1987 |
| 24 | Late romanticism orchestral | `Late_romanticism_orchestral_24` | 42 | 45.40 | 2170 |
| 25 | Latin-American | `Latin-American_25` | 30 | 28.91 | 958 |
| 26 | Middle, north European folklore | `Middle_north_European_folklore_26` | 40 | 10.09 | 431 |
| 27 | Middle, south Asian traditional folklore | `Middle_south_Asian_traditional_folklore_27` | 21 | 21.35 | 484 |
| 28 | North American traditional folklore | `North_American_traditional_folklore_28` | 9 | 4.50 | 431 |
| 29 | Pop | `Pop_29` | 47 | 58.54 | 509 |
| 30 | Pre-renaissance sacred | `Pre-renaissance_sacred_30` | 3 | 3.80 | 408 |
| 31 | Pre-renaissance secular | `Pre-renaissance_secular_31` | 7 | 7.16 | 180 |
| 32 | Rap, hip hop | `Rap_hip_hop_32` | 361 | 82.81 | 132 |
| 33 | Renaissance sacred | `Renaissance_sacred_33` | 6 | 6.53 | 696 |
| 34 | Renaissance secular | `Renaissance_secular_34` | 6 | 6.23 | 1029 |
| 35 | Rock | `Rock_35` | 508 | 163.22 | 258 |
| 36 | Romanticism chamber | `Romanticism_chamber_36` | 70 | 26.49 | 489 |
| 37 | Romanticism orchestral | `Romanticism_orchestral_37` | 12 | 10.46 | 217 |
| 38 | South American traditional folklore | `South_American_traditional_folklore_38` | 9 | 6.95 | 440 |
| 39 | South European folklore | `South_European_folklore_39` | 24 | 20.28 | 295 |
| 40 | Western songwriting | `Western_songwriting_40` | 251 | 131.26 | 175 |
| | **Total** | | **5414** | **1543.66** | **26538** |

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
| training (`train`) | 4323 (79.8%) | 1217.72 (78.9%) | 21229 (80.0%) |
| validation (`val`) | 542 (10.0%) | 151.99 (9.8%) | 2650 (10.0%) |
| test (`test`) | 549 (10.1%) | 173.95 (11.3%) | 2659 (10.0%) |
| total | 5414 | 1543.66 | 26538 |

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
| 18 | `Electronic_18` | 62 / 8 / 8 | 28.79 / 6.32 / 3.17 | 1280 / 160 / 160 |
| 19 | `Flemish_sacred_19` | 3 / 1 / 1 | 2.62 / 0.52 / 0.84 | 164 / 20 / 21 |
| 20 | `Flemish_secular_20` | 7 / 1 / 1 | 4.88 / 1.23 / 0.08 | 170 / 21 / 21 |
| 21 | `Heavy_metal_21` | 84 / 10 / 11 | 53.09 / 4.54 / 6.53 | 1064 / 133 / 133 |
| 22 | `Jazz_22` | 1111 / 139 / 139 | 168.51 / 17.26 / 21.25 | 243 / 30 / 30 |
| 23 | `Late_romanticism_chamber_23` | 67 / 8 / 8 | 32.54 / 1.77 / 3.98 | 1589 / 199 / 199 |
| 24 | `Late_romanticism_orchestral_24` | 34 / 4 / 4 | 40.20 / 2.45 / 2.75 | 1736 / 217 / 217 |
| 25 | `Latin-American_25` | 24 / 3 / 3 | 22.96 / 3.48 / 2.47 | 766 / 96 / 96 |
| 26 | `Middle_north_European_folklore_26` | 32 / 4 / 4 | 7.22 / 1.51 / 1.37 | 345 / 43 / 43 |
| 27 | `Middle_south_Asian_traditional_folklore_27` | 17 / 2 / 2 | 17.13 / 1.82 / 2.40 | 387 / 48 / 49 |
| 28 | `North_American_traditional_folklore_28` | 7 / 1 / 1 | 1.33 / 2.18 / 0.99 | 345 / 43 / 43 |
| 29 | `Pop_29` | 37 / 5 / 5 | 52.59 / 1.12 / 4.83 | 406 / 52 / 51 |
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
the performer or the composer; `date` is a year for 2056 tracks and a full
date or other text (such as `90s`) for 12.

Filled values, out of 5414 tracks:

| `info` | `title` | `artist` | `album` | `date` | `composer` |
|---:|---:|---:|---:|---:|---:|
| 5242 | 5171 | 4268 | 3405 | 2068 | 1570 |

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
| `file` | path of the image, from the dataset root |
| `class` | class name |
| `split` | `train`, `val` or `test`: the set of the file (the same as its folder) |
| `info` | artist, title and year read from the original file name, written as `Artist - Title (year)` or with the parts that are known; empty when the name carried no information (1673 images whose names carried no information, such as codes or hashes) |

```
file,class,split,info
Entire_images/train/Baroque_sacred_06/060769.jpg,Baroque sacred,train,Jacob Jordaens - Bust of Satyr (1621)
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
  these files). One of them, `Entire_music/test/Jazz_22/220745.mp3`, can be
  decoded only for its first 76.6 s, while its header states 304.0 s.
- `Entire_music/val/Late_romanticism_chamber_23/230007.m4a` and
  `Entire_music/train/Late_romanticism_chamber_23/230013.m4a` contain damaged AAC
  data: ffmpeg decodes them with many errors and ends with an error code;
  `preprocess_example.py` skips them.
- In `Entire_music/test/Pre-renaissance_sacred_30/300001.mp3` (1 h 43 min)
  the right channel is the left one with the opposite sign: each channel
  alone is normal, but their average (a mono mix) is silence.
  `preprocess_example.py` keeps the two channels apart.
- The header of some mp3 files states a wrong duration (one states 1058.5 s
  and contains 643.6 s). The hours in this README were measured by decoding.

## Images

| jpg | jpeg | png | webp |
|---:|---:|---:|---:|
| 26466 | 33 | 34 | 5 |

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
- Across classes, 1386 images appear as identical files in two or more
  classes (1405 extra copies), and one track appears in two classes
  (`Entire_music/train/Rap_hip_hop_32/320231.mp3` and
  `Entire_music/train/Rock_35/350287.mp3`). Each copy is a file of its own
  class, with its own number.
- 582 pairs of images inside a class, and 206 pairs across classes, are
  near-duplicates (perceptual hashes: pHash distance up to 8 and dHash
  distance up to 12), very probably the same artwork in two different
  reproductions. They were kept.
- 182 pairs of tracks of different length share part of their audio: a track
  and a longer file that contains it (a whole album in one file), an edit and
  its full version. 5 pairs of tracks of the same length share only part of
  their first 2 minutes (from 10 s to 79% of them). They were kept.

All the files with the same content are in the same set (training,
validation or test): 65 groups of tracks (251 tracks) and 1814 groups of
images (3798 images).

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
| `<split>/<class folder>/<number>_<hash>_ch<C>_<NNNN>.flac` | the segments, in the set of their track; `<number>` is the track (`350014`), `<hash>` 6 characters from its path, `<C>` the channel (`0` or `1`; `0` for mono files), `NNNN` the position of the segment in the track (`0000` = first 30 s after the leading silence) |
| `metadata.csv` | one row per segment: `filename`, `split`, `class_name`, `source_name`, `source_path` (the track, e.g. `val/Rock_35/350014.mp3`), `channel`, `seg_index`, `start_sec`, `duration_sec`, `gain_db` (the gain of the track), `mean_dbfs` (mean amplitude of the segment) |
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
