"""Check the MLX port against goldens.npz written by make_goldens.py (official PyTorch modules, fp32).

    python scripts/verify.py --goldens goldens.npz [--depth-dtype fp16|fp32] [--quantize 8]
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maya2_mlx.model import load_model  # noqa: E402
from maya2_mlx.prompt import Voice, build_prompt  # noqa: E402
from maya2_mlx.tokens import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--goldens", default="goldens.npz")
ap.add_argument("--model", default="maya-research/maya2-native")
ap.add_argument("--depth-dtype", default="fp16")
ap.add_argument("--quantize", type=int, default=None)
ap.add_argument("--skip-codec", action="store_true")
A = ap.parse_args()

g = np.load(A.goldens)
from huggingface_hub import snapshot_download  # noqa: E402

root = Path(snapshot_download(A.model, allow_patterns=["*.safetensors", "*.json", "backbone/*", "voices/*", "samples/*"]))
ok = True


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def check(name, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


# ---------------------------------------------------------------- tokenizer / prompt format
tok = Tokenizer(root / "backbone")
text = str(g["text"])
ids = [tok.lang_ids["hi"]] + tok.encode(f"{{Ritvik}}: {text}") + [tok.audio_id]
check("prompt ids", list(ids) == list(g["prompt_ids"]), f"{len(ids)} ids vs {len(g['prompt_ids'])}")
check("control ids", int(g["audio_id"]) == tok.audio_id and list(g["lang_ids"]) == [tok.lang_ids[l] for l in
      ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]], f"<|audio|>={tok.audio_id}")

from maya2_mlx.prompt import prompt_layout  # noqa: E402

cv = Voice("clone", "icl_spk", clips=[{"text": str(g["clone_ref_text"]), "lang": "hi", "codes": g["codes"][:, :20]}],
           ecapa=np.ones(192, np.float32) / np.sqrt(192))
toks, kinds = [], []
for kind, val in prompt_layout(tok, cv, str(g["clone_tgt_text"]), "ta"):
    if kind == "ids":
        toks += list(val); kinds += [0] * len(val)
    elif kind == "frames":
        toks += [None] * len(val); kinds += [1] * len(val)
    else:
        toks += [None]; kinds += [3]
gk, gt = list(g["clone_kind"]), list(g["clone_tokens"])
same = kinds == gk and all(t == u for t, u, k in zip(toks, gt, kinds) if k == 0)
check("clone prompt layout (cross-lingual icl_spk)", same, f"{len(kinds)} positions vs {len(gk)}")

# ---------------------------------------------------------------- model
dd = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[A.depth_dtype]
model = load_model(root, depth_dtype=dd, quantize_bits=A.quantize)
q = A.quantize is not None
emb, T = build_prompt(model, tok, Voice("Ritvik", "named"), text, "hi")
check("prompt embeddings", rel(emb[0].astype(mx.float32), g["prompt_emb"]) < 1e-2,
      f"max rel err {rel(emb[0].astype(mx.float32), g['prompt_emb']):.2e} (bf16 tables)")
codes = g["codes"]                       # (32, F)
nF = g["feedback"].shape[0]
F_all = codes.shape[1]
fb = model.audio_in.frames(mx.array(codes.T.astype(np.int32)))
check("feedback embeddings (eps " + str(g["feedback_eps"]) + ")",
      rel(fb[:nF].astype(mx.float32), g["feedback"]) < 2e-2, f"max rel err {rel(fb[:nF].astype(mx.float32), g['feedback']):.2e}")

cache = model.make_cache()
x = mx.concatenate([emb, fb[None].astype(emb.dtype)], axis=1)
hid = model.backbone(x, cache)[0]
h_pred = hid[T - 1:T - 1 + F_all]
lg = model.cb0_logits(h_pred)
mx.eval(lg)
lgn = np.array(lg)
nll = -(lgn - np.log(np.exp(lgn - lgn.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lgn.max(-1, keepdims=True))
nll = nll[np.arange(F_all), codes[0]]
gn = float(g["cb0_nll"].mean())
check("backbone hidden", rel(np.array(h_pred[:nF].astype(mx.float32)), g["hidden"]) < (0.15 if q else 0.06),
      f"max rel err {rel(np.array(h_pred[:nF].astype(mx.float32)), g['hidden']):.2e} (bf16 vs fp32 reference)")
check("cb0 NLL", abs(nll.mean() - gn) < (0.15 if q else 0.05),
      f"MLX {nll.mean():.3f} vs torch {gn:.3f} (uniform {np.log(2049):.2f}, wrong text {float(g['wrong_nll']):.3f})")
top1 = (lgn[:nF].argmax(-1) == g["cb0_logits"].argmax(-1)).mean()
check("cb0 top-1 agreement", top1 > (0.85 if q else 0.95), f"{top1:.1%}")

# ---------------------------------------------------------------- depth, teacher forced
table = model.depth.build_code_table()
cross = model.depth.cross_constants(model.depth.speaker_memory(None))
captured = []
forced = mx.array(codes.T[:nF].astype(np.int32))


def pick(level, logits):
    captured.append(logits)
    return forced[:, level]


model.depth.frame(h_pred[:nF], forced[:, 0], cross, table, pick)
dl = np.stack([np.array(c) for c in captured], 1)          # (nF, 31, 2048)
gd = g["depth_logits"]
err = rel(dl, gd)
lse = np.log(np.exp(dl - dl.max(-1, keepdims=True)).sum(-1)) + dl.max(-1)
d_nll = float((lse - np.take_along_axis(dl, codes.T[:nF, 1:, None], -1)[..., 0]).mean())
check("depth logits", err < (0.2 if q else 0.05), f"max rel err {err:.2e}")
check("depth NLL", abs(d_nll - float(g["depth_nll"])) < (0.15 if q else 0.05),
      f"MLX {d_nll:.3f} vs torch {float(g['depth_nll']):.3f}")
agree = (dl.argmax(-1) == gd.argmax(-1)).mean()
check("depth top-1 agreement", agree > (0.85 if q else 0.95), f"{agree:.1%}")

# ---------------------------------------------------------------- codec
if not A.skip_codec:
    from maya2_mlx.api import load_audio
    from maya2_mlx.codec import Codec

    codec = Codec()
    w = load_audio(root / "samples" / "hi_ritvik.wav")
    c2 = codec.encode(w)
    n = min(c2.shape[1], codes.shape[1])
    check("mimi encode (cb0 / all levels match transformers)", (c2[0, :n] == codes[0, :n]).mean() > 0.9,
          f"cb0 {(c2[0, :n] == codes[0, :n]).mean():.1%}, all {(c2[:, :n] == codes[:, :n]).mean():.1%}")
    rec = codec.decode(codes.T)
    r = g["recon"]
    m = min(len(rec), len(r))
    corr = float(np.corrcoef(rec[:m], r[:m])[0, 1])
    check("mimi decode vs transformers", corr > 0.98, f"waveform correlation {corr:.4f}")
    with codec.stream() as step:
        st = np.concatenate([step(f) for f in codes.T])
    corr2 = float(np.corrcoef(st[:m], rec[:m])[0, 1])
    check("mimi streaming == offline decode", corr2 > 0.99, f"waveform correlation {corr2:.4f}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
