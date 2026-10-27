"""Model-only correctness check (no reference code needed): teacher-force a release sample through the MLX port.

If the port is right, the codebook-0 loss of Maya's own sample under its own transcript and voice is far below
chance (ln 2049 = 7.63) and clearly lower than under a wrong transcript or a wrong voice; the depth loss over
codebooks 1-31 is far below chance too.

    python scripts/selfcheck.py [--quantize 8] [--codes codes.npy]
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maya2_mlx.model import load_model  # noqa: E402
from maya2_mlx.prompt import Voice, build_prompt  # noqa: E402
from maya2_mlx.text import normalise_text  # noqa: E402
from maya2_mlx.tokens import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="maya-research/maya2-native")
ap.add_argument("--quantize", type=int, default=None)
ap.add_argument("--depth-dtype", default="fp16")
ap.add_argument("--codes", default=None, help="precomputed (32, F) Mimi codes of samples/hi_ritvik.wav (.npy)")
A = ap.parse_args()

from huggingface_hub import snapshot_download  # noqa: E402

root = Path(snapshot_download(A.model, allow_patterns=["*.safetensors", "*.json", "backbone/*", "voices/*", "samples/*"]))
SAMPLE = "hi_ritvik.wav"
TEXT = "क्या आपका वर्तमान पता 45, एमजी रोड, अंधेरी ईस्ट, मुंबई 400069 है? यदि हाँ, तो 1 दबाएं।"
WRONG = "मौसम आज बहुत सुहावना है और हम शाम को पार्क में टहलने जाएंगे।"

if A.codes:
    codes = np.load(A.codes)
else:
    from maya2_mlx.api import load_audio
    from maya2_mlx.codec import Codec

    codes = Codec().encode(load_audio(root / "samples" / SAMPLE))
print("sample frames:", codes.shape[1])

t0 = time.perf_counter()
dd = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[A.depth_dtype]
model = load_model(root, depth_dtype=dd, quantize_bits=A.quantize)
tok = Tokenizer(root / "backbone")
print(f"model loaded in {time.perf_counter() - t0:.1f}s")
F = codes.shape[1]
fb = model.audio_in.frames(mx.array(codes.T.astype(np.int32)))


def cb0_nll(voice, text):
    emb, T = build_prompt(model, tok, Voice(voice, "named"), normalise_text(text, "hi"), "hi")
    h = model.backbone(mx.concatenate([emb, fb[None].astype(emb.dtype)], 1), model.make_cache())[0, T - 1:T - 1 + F]
    lg = model.cb0_logits(h)
    nll = mx.logsumexp(lg, -1) - mx.take_along_axis(lg, mx.array(codes[0].astype(np.int32))[:, None], -1)[:, 0]
    return float(nll.mean()), h


ok, h = cb0_nll("Ritvik", TEXT)
wrong_text, _ = cb0_nll("Ritvik", WRONG)
wrong_voice, _ = cb0_nll("Kayal", TEXT)
print(f"cb0 NLL  right text+voice {ok:.3f} | wrong text {wrong_text:.3f} | wrong voice (Kayal) {wrong_voice:.3f} "
      f"| chance {np.log(2049):.3f}")

table = model.depth.build_code_table()
cross = model.depth.cross_constants(model.depth.speaker_memory(None))
forced = mx.array(codes.T.astype(np.int32))
nlls = []


def pick(level, logits):
    t = forced[:, level]
    nlls.append(mx.logsumexp(logits, -1) - mx.take_along_axis(logits, t[:, None], -1)[:, 0])
    return t


model.depth.frame(h, forced[:, 0], cross, table, pick)
per_level = np.array([float(x.mean()) for x in nlls])
print(f"depth NLL mean {per_level.mean():.3f} (chance {np.log(2048):.3f}); levels 1/8/16/31: "
      f"{per_level[0]:.2f} {per_level[7]:.2f} {per_level[15]:.2f} {per_level[30]:.2f}")
good = ok < 4.0 and ok < wrong_text - 0.3 and per_level.mean() < 6.0
print("SELF-CHECK", "PASS" if good else "FAIL")
