"""Golden reference tensors from the official PyTorch modules (CPU, fp32).

Run once with a Python that has torch + transformers, pointing at a clone of
https://github.com/MayaResearch/maya2-native. The MLX port is checked against the
saved .npz by scripts/verify.py. Never load this and the MLX model in the same
process: the fp32 torch model alone is ~10 GB.

    python scripts/make_goldens.py --ref /path/to/maya2-native --out goldens.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="clone of github.com/MayaResearch/maya2-native")
ap.add_argument("--model", default="maya-research/maya2-native")
ap.add_argument("--out", default="goldens.npz")
ap.add_argument("--frames", type=int, default=40, help="teacher-forced frames of the sample to keep")
A = ap.parse_args()

sys.path.insert(0, str(Path(A.ref).resolve()))
torch.set_grad_enabled(False)

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MimiModel

from maya3.prompt import NamedSpeakerBuilder, Voice
from maya3.text import normalise_text
from tts.data.packing_v2 import PackerV2
from tts.model.anchors import SpeakerAnchor
from tts.model.audio_io import AudioInputEmbed
from tts.model.depth_v2 import DepthV2
from tts.tokens import extend_tokenizer

root = Path(snapshot_download(A.model, allow_patterns=["*.safetensors", "*.json", "backbone/*", "voices/*", "samples/*"]))

# ---------------------------------------------------------------- tokens / prompt
tok = AutoTokenizer.from_pretrained(root / "backbone")
n_before = len(tok)
tok, ids = extend_tokenizer(tok)
assert len(tok) == n_before, f"tokenizer grew {n_before} -> {len(tok)}: contract tokens were not in tokenizer.json"
packer = PackerV2(tok, ids)
voice = Voice("Ritvik", "named", builder=NamedSpeakerBuilder(), meta={"name": "Ritvik"})
raw = "क्या आपका वर्तमान पता 45, एमजी रोड, अंधेरी ईस्ट, मुंबई 400069 है? यदि हाँ, तो 1 दबाएं।"
text = normalise_text(raw, "hi")
rendered = voice.builder.render(voice, text)
cut = {"text_final": rendered, "lang_final": "hi", "codes": np.zeros((32, 1), np.int64), "row": 0}
t, k, c, m = packer._span(cut, first=True, cont=False)
prompt_ids = t[:-2]          # drop the dummy frame + AEOS: ends at <|audio|>
assert int(prompt_ids[-1]) == ids["<|audio|>"]
print("normalised:", text)
print("rendered  :", rendered)
print("prompt ids:", len(prompt_ids))

# ---------------------------------------------------------------- weights
idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
sd = {}
for shard in sorted(set(idx.values())):
    sd.update(load_file(str(root / shard)))
sd = {kk: v.float() for kk, v in sd.items()}
sd["backbone.lm.lm_head.weight"] = sd["backbone.lm.model.embed_tokens.weight"]

cfg = AutoConfig.from_pretrained(root / "backbone")
lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa").eval()
lm.load_state_dict({kk[len("backbone.lm."):]: v for kk, v in sd.items() if kk.startswith("backbone.lm.")}, strict=True)
cb0_head = torch.nn.Linear(2048, 2049, bias=False)
cb0_head.weight.copy_(sd["backbone.cb0_head.weight"])
audio_in = AudioInputEmbed(2048).eval()
audio_in.load_state_dict({kk[9:]: v for kk, v in sd.items() if kk.startswith("audio_in.")}, strict=True)
anchor = SpeakerAnchor(d_model=2048, input_dim=192, hidden=512).eval()
anchor.load_state_dict({kk[15:]: v for kk, v in sd.items() if kk.startswith("speaker_anchor.")}, strict=True)
depth = DepthV2(d_backbone=2048, grouped=False).eval()
depth.load_state_dict({kk[6:]: v for kk, v in sd.items() if kk.startswith("depth.")}, strict=True)
del sd

# ---------------------------------------------------------------- reference audio codes
import soundfile as sf
from scipy.signal import resample_poly

w, sr = sf.read(str(root / "samples" / "hi_ritvik.wav"), dtype="float32")
if w.ndim > 1:
    w = w.mean(1)
if sr != 24000:
    w = resample_poly(w, 24000, sr).astype(np.float32)
mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
codes = mimi.encode(torch.from_numpy(w)[None, None], num_quantizers=32).audio_codes[0]   # (32, F)
F_all = codes.shape[1]
recon = mimi.decode(codes[None]).audio_values[0, 0].numpy()
print("sample frames:", F_all)

# ---------------------------------------------------------------- compose + backbone
emb = lm.get_input_embeddings()(torch.as_tensor(prompt_ids)[None])                  # (1, T, D)
null = torch.tensor([True])
a_vec = anchor(torch.zeros(1, 1, 192), null_mask=null)                               # (1, D)
emb_anchor = emb.clone()
emb_anchor[0, -1] += a_vec[0]                                                        # target <|audio|> sentinel
# audio_in.norm is nn.RMSNorm(eps=None) -> eps = finfo(input dtype).eps: 1.2e-7 in fp32 but 2**-7 in the bf16
# serving engine. Which one the weights expect decides the feedback scale, so measure both.
T = len(prompt_ids)
e_pre = audio_in.table(codes.T + torch.arange(32) * 2048).sum(1) / np.sqrt(32)
print(f"feedback pre-norm mean square: median {e_pre.pow(2).mean(-1).median():.2e}")
results = {}
for name, eps in (("fp32", torch.finfo(torch.float32).eps), ("bf16", torch.finfo(torch.bfloat16).eps)):
    fb_e = torch.nn.functional.rms_norm(e_pre, (2048,), audio_in.norm.weight, eps=eps)
    hid = lm.model(inputs_embeds=torch.cat([emb_anchor, fb_e[None]], 1)).last_hidden_state[0]
    hp = hid[T - 1:T - 1 + F_all]
    lg = cb0_head(hp)
    nl = torch.nn.functional.cross_entropy(lg, codes[0], reduction="none")
    print(f"[eps={name}] cb0 NLL mean {nl.mean():.3f} (uniform = {np.log(2049):.3f})")
    results[name] = (fb_e, hp, lg, nl)
best = min(results, key=lambda kk: float(results[kk][3].mean()))
print("feedback eps that fits the weights:", best)
fb, h_pred, cb0_logits, nll = results[best]

# depth teacher-forced logits with the null speaker memory (named voice)
mem = depth.spk_memory(torch.zeros(1, 1, 192), "ecapa", null_mask=torch.tensor([True]))
nF = min(A.frames, F_all)
d_logits = depth(h_pred[:nF], codes.T[:nF], spk_mem=mem.expand(nF, -1, -1),
                 ref_mask=torch.ones(nF, 1, dtype=torch.bool), grouped=False)
d_logits = torch.stack(d_logits, 1)                                                  # (nF, 31, 2048)
d_nll = torch.nn.functional.cross_entropy(d_logits.reshape(-1, 2048), codes.T[:nF, 1:].reshape(-1))
print(f"depth NLL mean {d_nll:.3f}")

# wrong-text control: same audio, different sentence -> cb0 NLL must rise
wrong = voice.builder.render(voice, normalise_text("मौसम आज बहुत सुहावना है और हम पार्क में टहलने जाएंगे।", "hi"))
t2 = packer._span({"text_final": wrong, "lang_final": "hi", "codes": np.zeros((32, 1), np.int64), "row": 0},
                  first=True, cont=False)[0][:-2]
e2 = lm.get_input_embeddings()(torch.as_tensor(t2)[None]).clone()
e2[0, -1] += a_vec[0]
h2 = lm.model(inputs_embeds=torch.cat([e2, fb[None]], 1)).last_hidden_state[0]
nll2 = torch.nn.functional.cross_entropy(cb0_head(h2[len(t2) - 1:len(t2) - 1 + F_all]), codes[0])
print(f"cb0 NLL with WRONG text {nll2:.3f}")

# clone prompt layout (V5Builder, cross-lingual icl_spk: Hindi reference -> Tamil target), cut at the target sentinel
from maya3.prompt import V5Builder

ref_text = text
tgt_text = "வணக்கம், எப்படி இருக்கிறீர்கள்?"
cv = Voice("clone", "icl_spk", clips=[{"text": ref_text, "lang": "hi", "codes": codes.numpy()[:, :20]}],
           ecapa=np.ones(192, np.float32) / np.sqrt(192))
b5 = V5Builder()
cv.prefix = b5.prefix(packer, cv)
cv.spk_refs = cv.ecapa[None]
ex = b5.example(packer, cv, tgt_text, "ta")
ts = int(ex["target_start"])
a0 = ts + int(np.nonzero(np.asarray(ex["kind"])[ts:] == 1)[0][0])
clone_tokens, clone_kind = np.asarray(ex["tokens"])[:a0], np.asarray(ex["kind"])[:a0]
print("clone prompt:", a0, "positions, xling =", ex["xling"])

np.savez_compressed(
    A.out,
    clone_tokens=clone_tokens.astype(np.int64), clone_kind=clone_kind.astype(np.int64), clone_ref_text=np.array(ref_text),
    clone_tgt_text=np.array(tgt_text),
    prompt_ids=np.asarray(prompt_ids, np.int64), text=np.array(text), rendered=np.array(rendered),
    lang_ids=np.array([ids[f"<|lang:{l}|>"] for l in ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]]),
    audio_id=np.array(ids["<|audio|>"]),
    codes=codes.numpy().astype(np.int64), recon=recon.astype(np.float32),
    prompt_emb=emb_anchor[0].numpy(), anchor_null=a_vec[0].numpy(),
    feedback=fb[:nF].numpy(),
    hidden=h_pred[:nF].numpy(), cb0_logits=cb0_logits[:nF].numpy(), cb0_nll=nll.numpy(),
    depth_logits=d_logits.numpy().astype(np.float32), depth_nll=np.array(float(d_nll)),
    wrong_nll=np.array(float(nll2)), feedback_eps=np.array(best),
)
print("wrote", A.out)
