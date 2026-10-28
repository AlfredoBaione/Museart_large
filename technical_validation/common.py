"""
common.py -- paths, decoding of the music, ImageBind and CLAP shared by the scripts of the technical validation.
"""

import argparse
import csv
import math
import subprocess
import sys
import tempfile
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

SEGMENT_S, PIECE_S = 30, 10
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


def is_silent(samples):
    return 20 * math.log10(float(np.mean(np.abs(samples))) + 1e-7) < SILENCE_DB


def decode_stream(path, sample_rate):
    """FFmpeg decoding the first channel of the whole file to stdout (float32); stderr goes to a temporary file."""
    err = tempfile.TemporaryFile()
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-af", "pan=mono|c0=c0", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0), err


def read_samples(stream, n):
    buf = bytearray(4 * n)
    view, got = memoryview(buf), 0
    while got < len(buf):
        k = stream.readinto(view[got:])
        if not k:
            break
        got += k
    return np.frombuffer(buf, dtype=np.float32, count=got // 4)


def end_stream(process, err):
    """Returncode and messages of a decode_stream."""
    while process.stdout.read(1 << 16):
        pass
    rc = process.wait()
    err.seek(0)
    text = err.read().decode("utf-8", "replace").strip()
    err.close()
    return rc, text


def decode_piece(path, start, seconds, sample_rate):
    """First channel from `start`, resampled by FFmpeg; returns (samples, returncode, stderr)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{seconds:.3f}",
           "-af", "pan=mono|c0=c0", "-ar", str(sample_rate), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), p.returncode, p.stderr.decode("utf-8", "replace")


def load_imagebind(imagebind_dir, checkpoint, device="cuda", modalities=(VISION, AUDIO)):
    """ImageBind huge with only the given branches (vision, audio), loaded without pytorchvideo."""
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
    for modality in ("vision", "audio", "text", "depth", "thermal", "imu"):
        if modality in modalities:
            continue
        for modules in (model.modality_preprocessors, model.modality_trunks,
                        model.modality_heads, model.modality_postprocessors):
            del modules[modality]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    prefixes = tuple(f"modality_{part}.{m}." for part in ("preprocessors", "trunks", "heads", "postprocessors")
                     for m in modalities)
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
    if isinstance(input_features, np.ndarray):
        input_features, is_longer = torch.from_numpy(input_features), torch.from_numpy(is_longer)
    with torch.no_grad():
        out = model.get_audio_features(input_features=input_features.to(device), is_longer=is_longer.to(device))
    vectors = out.pooler_output if hasattr(out, "pooler_output") else out
    return torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()


class ClapFeaturesGPU:
    """The log-mel features of the CLAP feature extractor for pieces of exactly 10 s, computed with torch.stft in
    float64 (equal to those of the feature extractor); call load_clap first."""

    def __init__(self, device="cuda"):
        import torch
        from transformers.audio_utils import window_function
        assert _clap_features.truncation == "rand_trunc"
        assert _clap_features.nb_max_samples == PIECE_S * CLAP_SAMPLE_RATE
        self.n_fft, self.hop = _clap_features.fft_window_size, _clap_features.hop_length
        self.window = torch.from_numpy(window_function(self.n_fft, "hann")).to(device, torch.float64)
        self.filters = torch.from_numpy(_clap_features.mel_filters_slaney.T.copy()).to(device, torch.float64)

    def __call__(self, pieces):
        """[B, 480000] tensor on the device -> (input_features, is_longer)."""
        import torch
        spec = torch.stft(pieces.to(torch.float64), self.n_fft, self.hop, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        mel = torch.clamp(self.filters @ spec.abs() ** 2, min=1e-10)
        features = (10.0 * torch.log10(mel)).transpose(1, 2).float().unsqueeze(1)
        return features, torch.zeros((len(pieces), 1), dtype=torch.bool, device=pieces.device)
