"""
0_check_models.py -- a check of the two models before the runs, on the demo
files of the ImageBind repository (a dog, a car and a bird; no file of the
dataset is needed).

ImageBind: the vision x audio matrix of its README, with the audio decoded by
FFmpeg as in the other scripts. Expected (README of ImageBind):
    [[0.8070 0.1088 0.0842] [0.1036 0.7884 0.1079] [0.0018 0.0022 0.9960]]
CLAP: the three sounds against three text prompts ("a dog barking", "a car
engine", "a bird singing"); a working checkpoint puts most of them on the
right prompt.

Usage:
    python 0_check_models.py --imagebind ImageBind --checkpoint imagebind_huge.pth
"""

import wave

import numpy as np
import torch

import common

args = common.parse_args(__doc__.splitlines()[1], imagebind=True)
assets = args.imagebind / ".assets"
names = ["dog", "car", "bird"]
log = open(args.output / "check_models.log", "w", encoding="utf-8", buffering=1)


def say(line):
    print(line, flush=True)
    log.write(line + "\n")


def seconds(name):
    w = wave.open(str(assets / f"{name}_audio.wav"))
    return w.getnframes() / w.getframerate()


model = common.load_imagebind(args.imagebind, args.checkpoint)
images = torch.stack([common.load_image(assets / f"{n}_image.jpg") for n in names])
audio = torch.stack([common.piece_to_clips(torch.from_numpy(
    common.decode_piece(assets / f"{n}_audio.wav", 0, seconds(n), common.IB_SAMPLE_RATE)[0]).unsqueeze(0))
    for n in names])
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    vision = model({common.VISION: images.cuda()})[common.VISION].float().cpu()
    raw = model({common.AUDIO: audio.cuda()})[common.AUDIO].float().cpu()  # audio vectors carry the scale 20
say("ImageBind, vision x audio: " + np.array2string(torch.softmax(vision @ raw.T, dim=-1).numpy(),
                                                    precision=4).replace("\n", ""))
del model
torch.cuda.empty_cache()

clap = common.load_clap()
features = [common.clap_features(common.decode_piece(assets / f"{n}_audio.wav", 0, seconds(n),
                                                     common.CLAP_SAMPLE_RATE)[0]) for n in names]
sounds = common.embed_clap(clap, np.concatenate([f[0] for f in features]), np.concatenate([f[1] for f in features]))
from transformers import ClapProcessor  # noqa: E402
tokens = ClapProcessor.from_pretrained(common.CLAP_CHECKPOINT, revision=common.CLAP_REVISION)(
    text=["a dog barking", "a car engine", "a bird singing"], return_tensors="pt", padding=True)
with torch.no_grad():
    text = clap.get_text_features(**{k: v.cuda() for k, v in tokens.items()})
text = torch.nn.functional.normalize((text.pooler_output if hasattr(text, "pooler_output") else text).float(),
                                     dim=-1).cpu().numpy()
sim = sounds @ text.T
say(f"CLAP, sound x prompt cosine: {np.array2string(sim, precision=3)}".replace("\n", "")
    + f"; right prompt first for {int((sim.argmax(axis=1) == np.arange(3)).sum())} of 3")
log.close()
