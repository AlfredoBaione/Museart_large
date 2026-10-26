"""
common.py -- paths, pieces of music, ImageBind and CLAP shared by the scripts of the technical validation.
"""

import argparse
import csv
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_DIR = Path("Museart_Large")
OUTPUT_DIR = HERE / "output"
IMAGEBIND_DIR = Path("ImageBind")
IMAGEBIND_CHECKPOINT = Path("imagebind_huge.pth")
IMAGEBIND_COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
IMAGEBIND_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
IMAGEBIND_SHA256 = "d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157"
CLAP_CHECKPOINT = "laion/larger_clap_music_and_speech"
CLAP_REVISION = "195c3a3e68faebb3e2088b9a79e79b43ddbda76b"

STEP_S = 120.0       # one piece every 2 minutes of music
MIN_PIECES = 3
PIECE_S = 10.0
SILENCE_DB = -60.0   # mean amplitude (dBFS) of a silent piece

IB_SAMPLE_RATE = 16000   # ImageBind preprocessing, as imagebind/data.py
IB_CLIP_SECONDS = 2
IB_CLIPS_PER_PIECE = 3
IB_NUM_MEL_BINS = 128
IB_TARGET_LENGTH = 204
IB_AUDIO_MEAN, IB_AUDIO_STD = -4.268, 9.138
IB_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IB_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
VISION, AUDIO = "vision", "audio"

CLAP_SAMPLE_RATE = 48000


def parse_args(description, imagebind=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", type=Path, default=DATASET_DIR, help="folder of the dataset from Kaggle")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR, help="folder of the vectors and results")
    if imagebind:
        p.add_argument("--imagebind", type=Path, default=IMAGEBIND_DIR, help="clone of the ImageBind repository")
        p.add_argument("--checkpoint", type=Path, default=IMAGEBIND_CHECKPOINT, help="imagebind_huge.pth")
    args = p.parse_args()
    (args.output / "vectors").mkdir(parents=True, exist_ok=True)
    return args


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plan(seconds):
    """Start times of the pieces of a track: a piece centred in each of n equal parts."""
    n = max(MIN_PIECES, math.floor(seconds / STEP_S))
    last = max(seconds - PIECE_S, 0.0)
    return [min(max((i + 0.5) * seconds / n - PIECE_S / 2, 0.0), last) for i in range(n)]


def decode_piece(path, start, seconds, sample_rate):
    """First channel from `start`, resampled by FFmpeg; returns (samples, returncode, stderr)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{seconds:.3f}",
           "-af", "pan=mono|c0=c0", "-ar", str(sample_rate), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), p.returncode, p.stderr.decode("utf-8", "replace")


def load_imagebind(imagebind_dir, checkpoint, device="cuda"):
    """ImageBind huge with only the vision and audio branches, loaded without pytorchvideo."""
    import torch
    package = types.ModuleType("imagebind")
    package.__path__ = [str(Path(imagebind_dir) / "imagebind")]
    sys.modules["imagebind"] = package
    from imagebind.models import imagebind_model

    linspace = torch.linspace  # the drop-path rates are read with .item() while the model is built
    torch.linspace = lambda *args, **kwargs: linspace(*args, **{**kwargs, "device": "cpu"})
    try:
        with torch.device("meta"):
            model = imagebind_model.imagebind_huge(pretrained=False)
    finally:
        torch.linspace = linspace
    for modality in ("text", "depth", "thermal", "imu"):
        for modules in (model.modality_preprocessors, model.modality_trunks,
                        model.modality_heads, model.modality_postprocessors):
            del modules[modality]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    prefixes = tuple(f"modality_{part}.{m}." for part in ("preprocessors", "trunks", "heads", "postprocessors")
                     for m in (VISION, AUDIO))
    model.load_state_dict({k: v for k, v in state.items() if k.startswith(prefixes)}, strict=True, assign=True)
    on_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not on_meta, on_meta
    return model.eval().to(device)


_image_transform = None


def load_image(path):
    global _image_transform
    from PIL import Image
    from torchvision import transforms
    Image.MAX_IMAGE_PIXELS = None
    if _image_transform is None:
        _image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IB_IMAGE_MEAN, std=IB_IMAGE_STD),
        ])
    with open(path, "rb") as f:
        return _image_transform(Image.open(f).convert("RGB"))


def waveform2melspec(waveform):
    """As imagebind/data.py: [channels, samples] -> [1, 128, 204]."""
    import torch
    import torchaudio
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=IB_SAMPLE_RATE, use_energy=False, window_type="hanning",
        num_mel_bins=IB_NUM_MEL_BINS, dither=0.0, frame_length=25, frame_shift=10)
    fbank = fbank.transpose(0, 1)
    p = IB_TARGET_LENGTH - fbank.size(1)
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, :IB_TARGET_LENGTH]
    return fbank.unsqueeze(0)


def piece_to_clips(waveform):
    """[channels, samples] at 16 kHz -> [3, 1, 128, 204]: 3 clips of 2 s at uniform start times."""
    import torch
    seconds = waveform.size(1) / IB_SAMPLE_RATE
    max_start = max(seconds - IB_CLIP_SECONDS, 0)
    clips = []
    for i in range(IB_CLIPS_PER_PIECE):
        start = max_start * i / (IB_CLIPS_PER_PIECE - 1)
        clip = waveform[:, int(start * IB_SAMPLE_RATE):int((start + IB_CLIP_SECONDS) * IB_SAMPLE_RATE)]
        clips.append((waveform2melspec(clip) - IB_AUDIO_MEAN) / IB_AUDIO_STD)
    return torch.stack(clips)


def embed_imagebind(model, modality, batch, device="cuda"):
    """Images or pieces -> L2-normalized float32 array; a piece is the mean of its 3 clips."""
    import torch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model({modality: batch.to(device, non_blocking=True)})[modality]
    return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()


_clap_features = None


def load_clap(device="cuda"):
    global _clap_features
    from transformers import ClapFeatureExtractor, ClapModel
    _clap_features = ClapFeatureExtractor.from_pretrained(CLAP_CHECKPOINT, revision=CLAP_REVISION)
    return ClapModel.from_pretrained(CLAP_CHECKPOINT, revision=CLAP_REVISION).eval().to(device)


def clap_features(samples):
    """Samples at 48 kHz -> (input_features, is_longer) of the checkpoint; call load_clap first."""
    f = _clap_features(samples, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="np")
    return f["input_features"], f["is_longer"]


def embed_clap(model, input_features, is_longer, device="cuda"):
    """Batch of features -> L2-normalized float32 array [B, 512]."""
    import torch
    with torch.no_grad():
        out = model.get_audio_features(input_features=torch.from_numpy(input_features).to(device),
                                       is_longer=torch.from_numpy(is_longer).to(device))
    vectors = out.pooler_output if hasattr(out, "pooler_output") else out
    return torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()
