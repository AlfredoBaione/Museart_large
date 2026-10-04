"""
preprocess_example.py -- an example of how to prepare the music of the
dataset for training: every track becomes chunks of CHUNK_LENGTH_SEC seconds
(FLAC or WAV), kept in the train/val/test split of the track.

Input: Entire_music as released, already split into folders:
    Entire_music/train/<class>/..., Entire_music/val/<class>/...,
    Entire_music/test/<class>/...
(41 classes). Every track is trimmed of leading/trailing silence, brought to
a target loudness with one constant gain, split into its channels and cut
into chunks of CHUNK_LENGTH_SEC seconds.

1. THE SPLIT COMES WITH THE DATASET AND IS KEPT. Each chunk goes to the split
   of its track (OUTPUT/<split>/<class>/), so chunks of the same track -- and
   of its two channels -- never end up in different splits. The script does
   not split anything itself.

2. THE ACOUSTIC RULES (the same rules and values as `preprocess_stream.py
   --acoustic_rules` of our audio-generation pipeline):
     - leading and trailing silence removed: below SILENCE_TRIM_DB for at
       least 0.1 s (ffmpeg silencedetect);
     - loudness: ONE constant gain per track,
           gain_dB = min(TARGET_LUFS - measured loudness,
                         TARGET_TP - measured true peak),
       applied with ffmpeg `volume`. The track is never compressed and its
       true peak never goes above TARGET_TP. If the measurement fails, no
       gain is applied;
     - stereo: EACH CHANNEL becomes its own mono example (pan=mono|c0=cN),
       not the average of the two, which cancels out when the channels are in
       opposite phase (one track of the dataset becomes silence that way);
     - a chunk whose mean amplitude is below SILENCE_THRESH_DB dBFS is
       dropped as silence.

3. ONE ANALYSIS PASS AND ONE PASS PER CHANNEL, NO TEMPORARY FILES:
     - the analysis chains silencedetect and loudnorm (measuring only);
     - each channel pass trims, applies the gain, resamples to SR and streams
       raw 16-bit samples to this script, which cuts ALL the chunks by
       counting samples (exactly CHUNK_LENGTH_SEC each, one after the other)
       and writes them directly into the split folder.

   Note on the analysis: loudnorm measures the whole file, leading/trailing
   silence included. This does not bias the measurement, because EBU R128
   integrated loudness is gated: silence is already excluded from it.

4. NO FILES SKIPPED SILENTLY. The list of formats covers everything in
   Entire_music (ape, aiff, oma included), and at the end every file that was
   not used is printed -- and written to a CSV -- with the reason.

The GPU plays no role: audio decoding, silencedetect and loudnorm run on the
CPU only (ffmpeg accelerates video in hardware, not audio). The lever is
MAX_WORKERS.

Usage:
    python preprocess_example.py                 # uses the CONFIG below
    python preprocess_example.py --dry_run       # scan only, nothing is decoded
    python preprocess_example.py --workers 16    # overrides MAX_WORKERS
    python preprocess_example.py --format wav    # instead of flac
"""

import os
import re
import csv
import sys
import json
import math
import argparse
import hashlib
import threading
import subprocess
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

# =======================
# CONFIG
# =======================
SOURCE_DIR = "Entire_music"                # folder with train/, val/, test/
OUTPUT_DIR = "Entire_music_30sec_splits"   # output folder
CHUNK_LENGTH_SEC = 30                      # length of each chunk in seconds
SPLITS = ("train", "val", "test")          # split folders inside SOURCE_DIR
SR = 44100                                 # output sample rate

# Chunk format: "flac" or "wav". Exactly the same samples (FLAC is lossless,
# verified with np.array_equal), 16 bit in both cases.
#   flac -> measured on 8 genres of Entire_music: -43% of space
#           (471 GB -> ~267 GB), but reading costs 14 ms per 30-s chunk
#           instead of 2.2. On a slow or network disk, having half the bytes
#           to read pays for the decoding by itself.
#   wav  -> immediate reading, full space.
OUTPUT_FORMAT = "flac"

METADATA_FILE = "metadata.csv"
SKIPPED_FILE  = "skipped.csv"
MANIFEST_FILE = "manifest.jsonl"           # to resume an interrupted run

# --- Acoustic rules (values of preprocess_stream.py --acoustic_rules) ---
MIN_CHUNK_SEC     = CHUNK_LENGTH_SEC   # a track shorter than this after the
                                       # trim is skipped; a shorter tail is dropped
SILENCE_TRIM_DB   = -55.0              # leading/trailing silence threshold
SILENCE_THRESH_DB = -60.0              # mean amplitude (dBFS) below which a
                                       # chunk is silence
TARGET_LUFS       = -18.0              # loudness reached by the constant gain
TARGET_TP         = -1.0               # true peak never exceeded (dBTP)
TARGET_LRA        = 20.0               # only needed by loudnorm to measure
STEREO_SPLIT      = True               # each channel -> its own example;
                                       # False = average of the channels

# --- Performance ---
# Each track costs 1 + (number of channels) ffmpeg processes running in
# streaming mode: memory per worker is small and constant. On a dedicated
# machine you can go up to the number of cores. On a machine with 8 GB of RAM
# keep 2-3.
MAX_WORKERS = max(2, cpu_count() - 1)

# Accepted audio extensions. Everything in SOURCE_DIR that is not listed here
# ends up in the skipped.csv report, never silently dropped.
AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac",
    ".wma", ".mpc", ".ape", ".oma", ".aiff", ".aif", ".aifc", ".wv",
    ".alac", ".tta", ".ac3", ".dsf",
}
# Known extensions, deliberately ignored: they do not end up in the report as
# "unknown", because they are not audio and their absence is not a loss.
IGNORED_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".pdf",
    ".m3u", ".m3u8", ".pls", ".txt", ".nfo", ".log", ".cue",
    ".db", ".ini", ".sfv", ".doc", ".docx",
}


# =======================
# UTILITY
# =======================
def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Arbitrary file name -> filesystem-safe string (ASCII, lower case).
    """
    original_stem = Path(name).stem
    name = original_stem.lower()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if not name:
        return "unknown"
    if len(name) > max_length:
        short_hash = hashlib.md5(original_stem.encode("utf-8")).hexdigest()[:6]
        truncated = name[:max_length - 7].rsplit("_", 1)[0]
        name = f"{truncated}_{short_hash}"
    return name


def source_hash(rel_path: str) -> str:
    """6 characters from the SOURCE path: they make the name of every chunk
    unique even when two different tracks reduce to the same safe_name, and tie
    it to the file it comes from."""
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:6]


def probe_audio(file_path: str) -> tuple[float, str, int]:
    """
    Duration in seconds, codec name and number of channels of the first audio
    stream, via ffprobe (reads the headers, does not decode).
    (0.0, "", 0) if there is no audio.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,channels:format=duration",
                "-of", "default=noprint_wrappers=1:nokey=0",
                str(file_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        duration, codec, channels = 0.0, "", 0
        for line in (result.stdout or "").splitlines():
            if line.startswith("duration="):
                try:
                    duration = float(line.split("=", 1)[1])
                except ValueError:
                    duration = 0.0
            elif line.startswith("codec_name="):
                codec = line.split("=", 1)[1].strip()
            elif line.startswith("channels="):
                try:
                    channels = int(line.split("=", 1)[1])
                except ValueError:
                    channels = 0
        return duration, codec, channels
    except Exception:
        return 0.0, "", 0


def read_exactly(stream, n: int) -> bytes:
    """n bytes from a pipe; fewer only when the stream ends first."""
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            break
        buf += part
    return bytes(buf)


# =======================
# ANALYSIS: silence + loudness IN A SINGLE PASS
# =======================
def analyze_file(file_path: str, duration: float) -> dict:
    """
    A single decode of the track, with two chained filters:
      - silencedetect -> where the leading silence ends and the trailing one starts
      - loudnorm (print_format=json) -> measured integrated loudness and true
        peak (loudnorm only measures here: its output is thrown away)
    Both write to stderr, so they are read together.

    Returns {"trim_start", "trim_end", "gain_db" | None}, where
    gain_db = min(TARGET_LUFS - measured loudness, TARGET_TP - measured true
    peak): the constant gain of the track. None when the measurement failed
    (then no gain is applied).
    """
    out = {"trim_start": 0.0, "trim_end": duration, "gain_db": None}
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostdin",
                "-i", str(file_path),
                "-map", "0:a:0", "-vn",
                "-af", (
                    f"silencedetect=noise={SILENCE_TRIM_DB}dB:d=0.1,"
                    f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
                    f"print_format=json"
                ),
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception:
        return out

    stderr = result.stderr or ""

    # --- silencedetect: silence regions ---
    regions, current_start = [], None
    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                current_start = None
        elif "silence_end:" in line and current_start is not None:
            try:
                regions.append((current_start,
                                float(line.split("silence_end:")[1].strip().split()[0])))
            except (ValueError, IndexError):
                pass
            current_start = None
    if current_start is not None:            # silence that lasts until the end
        regions.append((current_start, duration))

    if regions:
        if regions[0][0] < 0.05:             # the silence starts at 0
            out["trim_start"] = regions[0][1]
        if regions[-1][1] >= duration - 0.05:  # and/or lasts until the end
            out["trim_end"] = regions[-1][0]

    # --- loudnorm: the JSON block is the last {...} of stderr ---
    js, je = stderr.rfind("{"), stderr.rfind("}") + 1
    if js != -1 and je > js:
        try:
            data = json.loads(stderr[js:je])
            mi = float(data.get("input_i", "nan"))
            mtp = float(data.get("input_tp", "nan"))
            if math.isfinite(mi) and math.isfinite(mtp):
                out["gain_db"] = min(TARGET_LUFS - mi, TARGET_TP - mtp)
        except Exception:
            out["gain_db"] = None

    return out


# =======================
# PER-TRACK WORK (worker): analysis + trim + normalize + cut
# =======================
def process_song(job: dict) -> dict:
    """
    One track -> its chunks, already written in the folder of its split.
    For a broken track it returns a dict with "status": skip, never a bare
    exception, because one broken file among 6000 must not stop the run.
    """
    import numpy as np
    import soundfile as sf

    src       = job["src"]
    rel_src   = job["rel_src"]
    cls       = job["class_name"]
    split     = job["split"]
    out_root  = Path(job["out_root"])

    def skipped(reason, detail=""):
        return {"status": "skip", "rel_src": rel_src, "class_name": cls,
                "reason": reason, "detail": str(detail)[:300]}

    duration, codec, n_channels = probe_audio(src)
    if duration <= 0.0 or not codec:
        return skipped("no readable audio stream", codec)
    if duration < MIN_CHUNK_SEC:
        return skipped("track shorter than one chunk", f"{duration:.1f}s")

    ana = analyze_file(src, duration)
    trim_start = max(0.0, ana["trim_start"])
    trim_end   = min(duration, ana["trim_end"])
    kept = trim_end - trim_start
    if kept < MIN_CHUNK_SEC:
        return skipped("too short after silence trimming", f"{kept:.1f}s")

    # One constant gain for the whole track (never a compressor): the same
    # value is applied to every channel.
    gain_db = ana["gain_db"]
    gain_filter = f"volume={gain_db:.2f}dB" if gain_db is not None else None
    # Stereo: one mono example per channel (channels 0 and 1). Averaging them
    # would cancel the music when the two channels are in opposite phase.
    channels = [0, 1] if (STEREO_SPLIT and n_channels >= 2) else [None]

    safe   = sanitize_filename(Path(src).name)
    shash  = source_hash(rel_src)
    outdir = out_root / split / cls
    outdir.mkdir(parents=True, exist_ok=True)
    # The format travels in the job, not in a global: on Windows the workers
    # are started with `spawn`, re-read the module from scratch and would never
    # see an OUTPUT_FORMAT changed from the command line inside main().
    ext    = job.get("fmt", OUTPUT_FORMAT)

    target_frames = int(round(CHUNK_LENGTH_SEC * SR))
    chunks, written, dropped_short, dropped_silent = [], [], 0, 0
    for ch in channels:
        filters = []
        if ch is not None:
            filters.append(f"pan=mono|c0=c{ch}")
        if gain_filter:
            filters.append(gain_filter)
        # One pass per channel: seek -> trim -> channel -> gain -> mono/SR,
        # written to a pipe as raw 16-bit samples. No temporary file, no
        # ffmpeg run per chunk.
        #
        # WHY THE CHUNKS ARE CUT HERE AND NOT BY FFMPEG: the `segment` muxer
        # cuts at the first packet whose timestamp reaches the next multiple
        # of CHUNK_LENGTH_SEC, so its segments last 30 s plus or minus one
        # packet (measured on an mp3: 30.015, 29.989, 30.015, 29.989... s),
        # and every segment even slightly shorter than 30 s had to be
        # discarded. Counting samples gives chunks of exactly
        # CHUNK_LENGTH_SEC, one after the other.
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-ss", str(trim_start),
            "-i", str(src),
            "-t", str(kept),
            "-map", "0:a:0", "-vn",
        ]
        if filters:
            command += ["-af", ",".join(filters)]
        command += ["-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le",
                    "-f", "s16le", "-"]
        channel = ch if ch is not None else 0
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # stderr is read in a thread: a long error log must not fill its pipe
        # and block ffmpeg while the samples are read from stdout.
        errors = []
        reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()))
        reader.start()
        try:
            seg_index = 0
            while True:
                data = read_exactly(proc.stdout, 2 * target_frames)
                if len(data) < 2 * target_frames:
                    if data:                     # the tail of the track
                        dropped_short += 1
                    break
                x = np.frombuffer(data, dtype="<i2")
                # Silence gate on the MEAN AMPLITUDE in dBFS (like ffmpeg
                # volumedetect's mean_volume), as in preprocess_stream.py.
                mean_dbfs = 20.0 * math.log10(float(np.mean(np.abs(x / 32768.0))) + 1e-7)
                if mean_dbfs < SILENCE_THRESH_DB:
                    dropped_silent += 1
                else:
                    p = outdir / f"{safe}_{shash}_ch{channel}_{seg_index:04d}.{ext}"
                    sf.write(str(p), x, SR, subtype="PCM_16")
                    written.append(p)
                    chunks.append({
                        "filename":     p.name,
                        "split":        split,
                        "class_name":   cls,
                        "source_name":  Path(src).name,
                        "source_path":  rel_src,
                        "channel":      channel,
                        "seg_index":    seg_index,
                        "start_sec":    round(trim_start + seg_index * CHUNK_LENGTH_SEC, 3),
                        "duration_sec": round(target_frames / float(SR), 3),
                        "gain_db":      "" if gain_db is None else round(gain_db, 2),
                        "mean_dbfs":    round(mean_dbfs, 2),
                        "bytes":        p.stat().st_size,   # not in the CSV: used by the report
                    })
                seg_index += 1
        except BaseException:
            proc.kill()
            for p in written:
                p.unlink(missing_ok=True)
            raise
        finally:
            proc.stdout.close()
            returncode = proc.wait()
            reader.join()

        if returncode != 0:
            # Nothing of a track that ffmpeg could not decode stays on disk.
            for p in written:
                p.unlink(missing_ok=True)
            msg = (errors[0] if errors else b"").decode("utf-8", errors="replace").strip()
            return skipped("ffmpeg could not decode it", msg)

    if not chunks:
        return skipped("no chunk survived the checks",
                       f"short={dropped_short} silent={dropped_silent}")

    return {"status": "ok", "rel_src": rel_src, "class_name": cls,
            "split": split, "chunks": chunks,
            "channels": len(channels),
            "dropped_short": dropped_short, "dropped_silent": dropped_silent,
            "gain_db": None if gain_db is None else round(gain_db, 2),
            "kept_sec": round(kept, 3)}


# =======================
# SOURCE SCAN
# =======================
def scan_source(source_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Scans SOURCE_DIR/<split>/<class>/** for every split in SPLITS. Returns
    (tracks, rejected). Recursive: classes may have subfolders (albums), and
    each track keeps its path relative to SOURCE_DIR as its identity. Files
    that are not inside a split and a class folder are reported, not used.
    """
    songs, rejected = [], []
    for split in SPLITS:
        if not (source_dir / split).is_dir():
            print(f"  WARNING: no '{split}' folder in {source_dir}")
    for fp in sorted(source_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(source_dir).as_posix()
        parts = rel.split("/")
        ext = fp.suffix.lower()
        if ext in IGNORED_EXTS:
            continue
        if len(parts) < 3 or parts[0] not in SPLITS:
            rejected.append({"rel_src": rel, "class_name": "",
                             "reason": "not inside <split>/<class>/",
                             "detail": parts[0]})
        elif ext in AUDIO_EXTS:
            songs.append({"src": str(fp), "rel_src": rel,
                          "split": parts[0], "class_name": parts[1]})
        else:
            rejected.append({"rel_src": rel, "class_name": parts[1],
                             "reason": "unrecognized extension",
                             "detail": ext or "(none)"})
    return songs, rejected


# =======================
# MANIFEST (resuming an interrupted run)
# =======================
def load_manifest(out_root: Path) -> dict:
    """
    rel_src -> record, for every track already EXAMINED in a previous run,
    whether it succeeded or was skipped.

    Skipped tracks go into the manifest too, for two reasons: a decode that
    has already failed is not retried at every restart (on 6000 files resuming
    must be cheap), and skipped.csv stays complete instead of containing only
    the skips of the last run. To really retry them: --force.
    """
    path = out_root / MANIFEST_FILE
    done = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["rel_src"]] = rec
            except Exception:
                continue
    return done


# =======================
# MAIN
# =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE_DIR)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--format", choices=("flac", "wav"), default=OUTPUT_FORMAT,
                    help=f"Chunk format (default: {OUTPUT_FORMAT}). "
                         f"Same samples, flac takes ~43%% less space.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Scan only (tracks per split and class), without "
                         "decoding anything.")
    ap.add_argument("--force", action="store_true",
                    help="Also reprocess the tracks already in the manifest.")
    args = ap.parse_args()

    src_root = Path(args.source)
    out_root = Path(args.output)
    if not src_root.is_dir():
        sys.exit(f"[error] the source folder does not exist: {src_root}")

    # --- 1. scan ---
    print(f"Step 1/3 - Scanning {src_root} ...")
    songs, rejected = scan_source(src_root)
    per_class = Counter(s["class_name"] for s in songs)
    per_split = Counter(s["split"] for s in songs)
    per_ext = Counter(Path(s["rel_src"]).suffix.lower() for s in songs)
    print(f"  {len(songs)} audio files in {len(per_class)} classes")
    print("  per split: " + ", ".join(f"{k}: {per_split.get(k, 0)}" for k in SPLITS))
    print("  formats: " + ", ".join(f"{e.lstrip('.')}:{n}"
                                    for e, n in per_ext.most_common()))
    if rejected:
        print(f"  {len(rejected)} files NOT used (they will go to {SKIPPED_FILE})")

    done = {} if args.force else load_manifest(out_root)
    todo = [s for s in songs if s["rel_src"] not in done]
    if done:
        n_ok = sum(1 for r in done.values() if r.get("status") == "ok")
        print(f"  {len(done)} tracks already examined in a previous run "
              f"({n_ok} succeeded, {len(done) - n_ok} skipped), "
              f"{len(todo)} remaining")

    if args.dry_run:
        print("\n--dry_run: stopping here, nothing was decoded.")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    for s in todo:
        s["out_root"] = str(out_root)
        s["fmt"] = args.format

    # --- 2. processing ---
    print(f"\nStep 2/3 - Trim + normalization + cutting ({args.workers} workers)...")
    print(f"  output format: {args.format}")
    print(f"  stereo split: {'each channel -> its own example' if STEREO_SPLIT else 'average of the channels'}; "
          f"gain to {TARGET_LUFS} LUFS (true peak <= {TARGET_TP} dBTP), "
          f"trim below {SILENCE_TRIM_DB} dB, chunks below {SILENCE_THRESH_DB} dBFS dropped")
    print("  1 analysis pass + 1 pass per channel, streaming: no temporary files on disk.")
    # Rebuild the state of previous runs from the manifest: the chunks already
    # written and the tracks already skipped. This way metadata.csv and
    # skipped.csv are always the complete picture, not just the last piece of
    # work.
    all_chunks = [c for rec in done.values() for c in rec.get("chunks", [])]
    skipped = list(rejected)
    skipped += [rec for rec in done.values() if rec.get("status") == "skip"]

    manifest_f = open(out_root / MANIFEST_FILE,
                      "w" if args.force else "a", encoding="utf-8")
    try:
        with Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_song, todo),
                            total=len(todo), desc="Tracks"):
                if res["status"] == "ok":
                    all_chunks.extend(res["chunks"])
                else:
                    skipped.append(res)
                manifest_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                manifest_f.flush()
    finally:
        manifest_f.close()

    # --- 3. report ---
    print("\nStep 3/3 - Metadata and report...")
    meta_fields = ["filename", "split", "class_name", "source_name",
                   "source_path", "channel", "seg_index", "start_sec",
                   "duration_sec", "gain_db", "mean_dbfs"]
    with open(out_root / METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_fields)
        w.writeheader()
        for c in sorted(all_chunks, key=lambda x: (x["class_name"],
                                                   x["source_path"],
                                                   x["channel"],
                                                   x["seg_index"])):
            w.writerow({k: c[k] for k in meta_fields})

    with open(out_root / SKIPPED_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rel_src", "class_name",
                                          "reason", "detail"])
        w.writeheader()
        for s in sorted(skipped, key=lambda x: (x.get("reason", ""),
                                                x.get("rel_src", ""))):
            w.writerow({k: s.get(k, "") for k in
                        ["rel_src", "class_name", "reason", "detail"]})

    split_counts = Counter(c["split"] for c in all_chunks)
    # Real space used, not estimated: with OUTPUT_FORMAT="flac" it depends on
    # the material (orchestral music compresses to 40%, metal to 68%).
    used = sum(c.get("bytes", 0) for c in all_chunks) / (1024 ** 3)
    raw = len(all_chunks) * CHUNK_LENGTH_SEC * SR * 2 / (1024 ** 3)
    print("\n" + "=" * 52)
    print("COMPLETED")
    print("=" * 52)
    if args.format == "flac" and used > 0 and raw > 0:
        print(f"  chunks written:  {len(all_chunks)}  ({used:.1f} GB in flac, "
              f"{raw:.1f} GB if they were wav -> -{100 * (1 - used / raw):.0f}%)")
    else:
        print(f"  chunks written:  {len(all_chunks)}  ({used or raw:.1f} GB)")
    for s in ("train", "val", "test"):
        print(f"    {s:<5} {split_counts.get(s, 0)}")
    print(f"  tracks not used: {len(skipped)}  -> {out_root / SKIPPED_FILE}")
    if skipped:
        for reason, n in Counter(s.get("reason", "?") for s in skipped).most_common():
            print(f"    {n:>5}  {reason}")
    print(f"  metadata:  {out_root / METADATA_FILE}")


if __name__ == "__main__":
    main()
