# Museart_2

**[TODO (authors): two or three sentences on what the dataset is for, how the
images were chosen and associated with the music classes, and where the
recordings and the images come from.]**

Museart_2 contains music recordings and images of artworks, organized in the
same 41 classes (genres, historical periods and traditions of music, from
*Pre-renaissance sacred* to *Heavy metal*):

- **5988 full-length tracks**, 1577.91 hours, in their original formats
  (128.9 GiB);
- **26498 images** (3.7 GiB);
- two CSV files with the class and the available information for every file.

| | Where |
|---|---|
| Data | Kaggle: **[TODO: link]** |
| Preprocessing code (`creation_v2.py`) | GitHub: **[TODO: link]** |
| Paper | **[TODO: citation]** |

## Contents

```
Entire_music/          41 class folders, 5988 audio files
Entire_images/         41 class folders, 26498 images
metadata_music.csv     one row per audio file
metadata_images.csv    one row per image
```

The files are released as they are: no format conversion, resampling,
loudness normalization or cutting. The 30-second segments and the
train/validation/test split are made by `creation_v2.py` (see
[Preparing the audio](#preparing-the-audio-creation_v2py)).

### File and folder names

Each class has a two-digit code, `00` to `40` (classes in alphabetical order),
and a folder with the same name in both sets: the class name with spaces
replaced by `_`, commas removed, and the code at the end (`Rock_35`,
`Rap_hip_hop_32`). Each file is named with the class code and a four-digit
number counted from `0001` inside its folder, followed by the original
extension in lower case:

```
Entire_music/Rock_35/350014.mp3
Entire_images/Baroque_sacred_06/060632.jpg
```

Audio files and images are numbered separately:
`Entire_music/Rock_35/350014.mp3` and `Entire_images/Rock_35/350014.jpg` are
not related.

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
| 06 | Baroque sacred | `Baroque_sacred_06` | 129 | 25.12 | 701 |
| 07 | Baroque secular | `Baroque_secular_07` | 1501 | 94.58 | 868 |
| 08 | Classical sacred | `Classical_sacred_08` | 5 | 6.37 | 553 |
| 09 | Classical secular | `Classical_secular_09` | 62 | 24.39 | 405 |
| 10 | Country | `Country_10` | 11 | 14.19 | 111 |
| 11 | Dance | `Dance_11` | 365 | 112.22 | 214 |
| 12 | Early 20th century experimental chamber | `Early_20th_century_experimental_chamber_12` | 8 | 5.22 | 490 |
| 13 | Early 20th century experimental orchestral | `Early_20th_century_experimental_orchestral_13` | 37 | 18.68 | 476 |
| 14 | Early 20th century traditional chamber | `Early_20th_century_traditional_chamber_14` | 80 | 32.29 | 874 |
| 15 | Early 20th century traditional orchestral | `Early_20th_century_traditional_orchestral_15` | 31 | 32.00 | 922 |
| 16 | East Asian traditional folklore | `East_Asian_traditional_folklore_16` | 18 | 18.76 | 844 |
| 17 | East European folklore | `East_European_folklore_17` | 24 | 18.15 | 534 |
| 18 | Electronic | `Electronic_18` | 77 | 38.20 | 1600 |
| 19 | Flemish sacred | `Flemish_sacred_19` | 5 | 3.98 | 362 |
| 20 | Flemish secular | `Flemish_secular_20` | 9 | 6.19 | 370 |
| 21 | Heavy metal | `Heavy_metal_21` | 111 | 64.59 | 1330 |
| 22 | Jazz | `Jazz_22` | 1518 | 221.69 | 303 |
| 23 | Late romanticism chamber | `Late_romanticism_chamber_23` | 19 | 13.93 | 699 |
| 24 | Late romanticism orchestral | `Late_romanticism_orchestral_24` | 17 | 19.81 | 1133 |
| 25 | Latin-American | `Latin-American_25` | 25 | 24.50 | 857 |
| 26 | Middle, north European folklore | `Middle_north_European_folklore_26` | 45 | 11.62 | 431 |
| 27 | Middle, south Asian traditional folklore | `Middle_south_Asian_traditional_folklore_27` | 23 | 23.26 | 490 |
| 28 | North American traditional folklore | `North_American_traditional_folklore_28` | 3 | 4.08 | 433 |
| 29 | Pop | `Pop_29` | 44 | 55.41 | 509 |
| 30 | Pre-renaissance sacred | `Pre-renaissance_sacred_30` | 4 | 5.04 | 399 |
| 31 | Pre-renaissance secular | `Pre-renaissance_secular_31` | 7 | 7.66 | 187 |
| 32 | Rap, hip hop | `Rap_hip_hop_32` | 373 | 83.77 | 132 |
| 33 | Renaissance sacred | `Renaissance_sacred_33` | 6 | 6.19 | 508 |
| 34 | Renaissance secular | `Renaissance_secular_34` | 25 | 7.99 | 685 |
| 35 | Rock | `Rock_35` | 583 | 167.84 | 258 |
| 36 | Romanticism chamber | `Romanticism_chamber_36` | 75 | 29.04 | 687 |
| 37 | Romanticism orchestral | `Romanticism_orchestral_37` | 15 | 13.13 | 459 |
| 38 | South American traditional folklore | `South_American_traditional_folklore_38` | 14 | 11.36 | 541 |
| 39 | South European folklore | `South_European_folklore_39` | 28 | 21.17 | 295 |
| 40 | Western songwriting | `Western_songwriting_40` | 260 | 130.75 | 175 |
| | **Total** | | **5988** | **1577.91** | **26498** |

## metadata_music.csv

One row per audio file. UTF-8, comma-separated (values containing a comma are
in double quotes), Windows line ends (CRLF).

| Column | Content |
|---|---|
| `file` | path of the audio file, from the dataset root |
| `class` | class name |
| `info` | the original file name without extension. Names that were hard to read were tidied (emoji, decorative letters, underscores used as spaces), signatures and web addresses were removed, unreadable names were rebuilt from the tags, and names with no information were left empty |
| `title`, `artist`, `album`, `date`, `composer` | from the tags of the original file. Text that is not about the music was removed (signatures, web addresses, values written by software such as `Unknown Artist` or `Track 5`, the name of the uploader and the upload date), and broken accents were repaired. Empty fields were then filled, where possible, with what the original file name says |

Values are written as they were found: in classical music `artist` can hold
the performer or the composer; `date` is a year for 2403 tracks and a full
date or other text (such as `90s`) for 12.

Filled values, out of 5988 tracks:

| `info` | `title` | `artist` | `album` | `date` | `composer` |
|---:|---:|---:|---:|---:|---:|
| 5803 | 5691 | 4810 | 3911 | 2415 | 1749 |

174 tracks have none of the five tag fields; 185 have an empty `info`.

```
file,class,info,title,artist,album,date,composer
Entire_music/Rock_35/350014.mp3,Rock,01-Hells Bells,Hells Bells,AC/DC,Back In Black,1980,
Entire_music/Baroque_sacred_06/060001.mp3,Baroque sacred,"01 - Prelude in E Flat Major, BWV 552","Prelude in E Flat Major, BWV 552",Wolfgang Ruebsam,"Clavierübung III, Vol.1 - J.S.Bach",1994,Johann Sebastian Bach
Entire_music/Baroque_sacred_06/060003.mp3,Baroque sacred,,,,,,
```

## metadata_images.csv

One row per image, same format as above.

| Column | Content |
|---|---|
| `file` | path of the image, from the dataset root |
| `class` | class name |
| `info` | artist, title and year read from the original file name, written as `Artist - Title (year)` or with the parts that are known; empty when the name carried no information (1660 images whose names were codes or hashes) |

```
file,class,info
Entire_images/Baroque_sacred_06/060632.jpg,Baroque sacred,Jacob Jordaens - Bust of Satyr (1621)
Entire_images/Electronic_18/180321.jpg,Electronic,Andy Warhol
Entire_images/Heavy_metal_21/210001.jpg,Heavy metal,
```

## Audio files

| mp3 | wma | m4a | mpc | ape | ogg | flac | wav | oma | aiff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5624 | 116 | 61 | 61 | 54 | 17 | 16 | 14 | 14 | 11 |

The audio data are those of the original files. The tags were rewritten
without re-encoding: inside each file they hold the same five values as the
CSV (`title`, `artist`, `album`, `date`, `composer`, where the format allows
it), and nothing else apart from technical fields used by decoders (such as
the encoder delay of m4a files). Covers, comments, genres, track numbers and
all other tags were removed.

Known problems, measured with ffmpeg:

- 302 files contain damaged frames, which decoders skip (58.75 hours in
  these files). One of them, `Jazz_22/220804.mp3`, can be decoded only for
  its first 76.6 s, while its header states 304.0 s.
- `Romanticism_chamber_36/360017.m4a` and `Romanticism_chamber_36/360020.m4a`
  contain damaged AAC data: ffmpeg decodes them with many errors and ends
  with an error code; `creation_v2.py` skips them.
- The header of some mp3 files states a wrong duration (one states 1647.8 s
  and contains 499.9 s). The hours in this README were measured by decoding.

## Images

| jpg | jpeg | png |
|---:|---:|---:|
| 26443 | 31 | 24 |

The image files are the original ones, byte for byte.

## Duplicates

- Inside a class there are no identical files (images: identical bytes;
  audio: identical audio data).
- Across classes, 1548 images appear as identical files in two or more
  classes (1569 extra copies), and one track appears in two classes
  (`Rap_hip_hop_32/320236.mp3` and `Rock_35/350334.mp3`). Each copy is a file
  of its own class, with its own number.
- Inside a class, 594 pairs of images are near-duplicates (perceptual hashes:
  pHash distance up to 8 and dHash distance up to 12), very probably the same
  artwork in two different reproductions. They were kept.

When the images are split into training and test sets, copies of the same
artwork can end up on both sides.

## Preparing the audio: creation_v2.py

`creation_v2.py` (GitHub: **[TODO: link]**) turns `Entire_music` into
segments of 30 seconds, split into training, validation and test sets.

### Requirements

- Python 3.9 or later, with numpy, soundfile and tqdm:
  `pip install numpy soundfile tqdm`;
- `ffmpeg` and `ffprobe` on the PATH.

Tested on Windows 11 with Python 3.14, numpy 2.4, soundfile 0.13, tqdm 4.67
and an ffmpeg build of March 2026.

### Usage

```
python creation_v2.py --source path/to/Entire_music --output path/to/Entire_music_30sec_splits --workers 8
```

| Option | Meaning |
|---|---|
| `--source` | the `Entire_music` folder (default: `Entire_music`) |
| `--output` | output folder (default: `Entire_music_30sec_splits`) |
| `--workers` | tracks processed in parallel (default: number of CPU cores minus one, at least 2) |
| `--format` | `flac` (default) or `wav`: the same samples, FLAC takes about half the space |
| `--dry_run` | only list the tracks and assign the split (written to `splits.json`); nothing is decoded |
| `--force` | process again the tracks already done |

The work is done by ffmpeg on the CPU (the GPU is not used). An interrupted
run continues where it stopped when the same command is run again
(`manifest.jsonl` records the tracks already done).

### What it does to each track

1. One pass with ffmpeg finds the leading and trailing silence (below -35 dB
   for at least 0.1 s) and measures the loudness (EBU R128).
2. A second pass removes that silence, normalizes the loudness to -14 LUFS
   (true peak -1 dBTP, loudness range 11 LU; two-pass `loudnorm` filter,
   linear gain when possible, otherwise its dynamic mode) and converts the
   audio to mono, 44 100 Hz, 16 bit.
3. The track is cut into consecutive segments of exactly 30 s
   (1 323 000 samples). The last part, shorter than 30 s, is discarded, and so
   is every segment whose RMS level is below -40 dBFS (silence).

Tracks shorter than 30 s (also after the silence is removed) and files that
ffmpeg cannot decode are listed in `skipped.csv` with the reason.

The output has fewer than 190 000 segments (1577.91 hours / 30 s); a 30-s
WAV segment takes 2.6 MB, so the WAV output needs up to about 500 GB, and the
FLAC output about half of that.

### Output

| Path | Content |
|---|---|
| `<split>/<class folder>/<number>_<hash>_<NNNN>.flac` | the segments; `<number>` is the track (`350014`), `<hash>` 6 characters from its path, `NNNN` the position of the segment in the track (`0000` = first 30 s after the leading silence) |
| `metadata.csv` | one row per segment: `filename`, `split`, `class_name`, `source_name`, `source_path`, `seg_index`, `start_sec`, `duration_sec`, `rms_db` |
| `splits.json` | the split of every track |
| `skipped.csv` | the tracks not used, with the reason |
| `manifest.jsonl` | progress of the run |

The class of a segment (folders and `class_name`) is the folder name
(`Rock_35`); `metadata_music.csv` gives the class name of every folder.

### Training, validation and test split

- **By track:** all the segments of a track are in the same set, so no track
  is in two sets.
- 80% / 10% / 10% of the tracks of every class (stratified by class); every
  class with at least 3 tracks has at least one track in validation and one
  in test.
- Fixed seed (42). Run on the complete `Entire_music` folder, the script
  gives this split, the same at every run:

| Set | Tracks | Hours |
|---|---:|---:|
| training | 4782 | 1248.56 |
| validation | 597 | 165.58 |
| test | 609 | 163.77 |

The split is computed from the list of files: on only part of the files it is
different. It is written to `splits.json` at the first run (also with
`--dry_run`) and reused by the later runs in the same output folder, so start
from an empty output folder to obtain the split above.

## License

- Code (`creation_v2.py`): MIT License, see `LICENSE` in the GitHub
  repository.
- Data: **[TODO (authors): licence of the data.]**

## Citation

**[TODO (authors): authors, contact and citation of the paper (BibTeX).]**
