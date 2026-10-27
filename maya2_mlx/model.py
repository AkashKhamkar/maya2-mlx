"""Maya 2 Native in MLX.

Architecture (ported from github.com/MayaResearch/maya2-native, Apache-2.0):

  backbone   Qwen3-1.7B (mlx-lm's Qwen3Model, fed with input embeddings) + cb0 head (2048 codes + <aeos>)
  audio_in   backbone-width embedding of a 32-codebook Mimi frame (sum of per-level rows / sqrt(32), RMSNorm)
  anchor     ECAPA-192 (or a learned null vector) -> MLP -> added at every span's <|audio|> sentinel
  depth      DepthV2: predicts codebooks 1..31 of a frame from the backbone hidden state and cb0.
             11 weight groups x 4 pre-norm layers (1024 wide, 16 heads, GELU FFN 3072) + 4 speaker
             cross-attention layers shared across groups. Levels 1-8 own groups 0-7; levels 9-16 /
             17-24 / 25-31 share groups 8 / 9 / 10.

Two exact simplifications used at inference (same as the reference fast path, serve/fast_depth.py):
  * one speaker-memory token => cross-attention softmax is identically 1, so each cross layer adds the
    constant o(v(mem)) independent of the query;
  * within a shared band group the trunk is causal and position k is embedded from slot k only, so the
    K/V of earlier positions are reused and each later level costs one new position.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .tokens import AEOS_CLASS, N_CODES, N_Q

D, H, FFN, LAYERS, RANK = 1024, 16, 3072, 4, 128
HEAD = D // H
N_GROUPS = 11
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7
# audio_in.norm is torch nn.RMSNorm(eps=None): torch takes eps from the fp32 *compute* type even for bf16 inputs.
# Measured on a release sample: cb0 NLL 1.11 with this eps vs 1.88 with the bf16 eps (2**-7).
RMS_EPS = 1.1920928955078125e-07


# ----------------------------------------------------------------------------- backbone-side modules
class AudioInputEmbed(nn.Module):
    def __init__(self, d_backbone=2048, eps=RMS_EPS):
        super().__init__()
        self.table = nn.Embedding(N_Q * N_CODES, d_backbone)
        self.aeos_in = mx.zeros((d_backbone,))
        self.norm = nn.RMSNorm(d_backbone, eps=eps)
        self._offsets = mx.arange(N_Q) * N_CODES

    def frames(self, codes):
        """codes (..., 32) int -> (..., d_backbone)."""
        e = self.table(codes + self._offsets).astype(mx.float32).sum(-2) / math.sqrt(N_Q)
        return self.norm(e.astype(self.table.weight.dtype))

    def aeos(self):
        return self.norm(self.aeos_in)


class SpeakerAnchor(nn.Module):
    def __init__(self, d_model=2048, input_dim=192, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.null_vec = mx.zeros((input_dim,))

    def __call__(self, ecapa=None):
        """ecapa (192,) L2-normalised, or None for the learned null input. Returns (d_model,)."""
        x = self.null_vec if ecapa is None else ecapa
        x = x.astype(mx.float32)
        return self.fc2(nn.gelu(self.fc1(x)))      # exact (erf) GELU, as torch's default


# ----------------------------------------------------------------------------- depth decoder
class _SelfBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.f1 = nn.Linear(D, FFN, bias=False)
        self.f2 = nn.Linear(FFN, D, bias=False)


class _CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)


def _heads(x):            # (B, T, D) -> (B, H, T, HEAD)
    B, T, _ = x.shape
    return x.reshape(B, T, H, HEAD).transpose(0, 2, 1, 3)


class DepthDecoder(nn.Module):
    def __init__(self, d_backbone=2048):
        super().__init__()
        self.in_proj = nn.Linear(d_backbone, D, bias=False)
        self.code_lo = nn.Embedding((N_Q - 1) * N_CODES, RANK)
        self.code_up = nn.Linear(RANK, D, bias=False)
        self.level_emb = nn.Embedding(N_Q, D)
        self.groups = [[_SelfBlock() for _ in range(LAYERS)] for _ in range(N_GROUPS)]
        self.cross = [_CrossBlock() for _ in range(LAYERS)]
        self.norm = nn.LayerNorm(D)
        self.heads = [nn.Linear(D, N_CODES, bias=False) for _ in range(N_Q - 1)]
        self.spk_proj_campp = nn.Linear(1024, D, bias=False)
        self.spk_proj_ecapa = nn.Linear(192, D, bias=False)
        self.null_mem = mx.zeros((1, 1, D))
        self.scale = HEAD ** -0.5

    # --- inference-time constants
    def build_code_table(self):
        """(31, 2048, D): prefix embedding of code c at slot k = code_up(code_lo[c + k*2048]) + level_emb[k+1]."""
        t = self.code_up(self.code_lo.weight).reshape(N_Q - 1, N_CODES, D)
        return t + self.level_emb.weight[1:N_Q, None]

    def speaker_memory(self, ecapa=None):
        """(1, D) memory token: projected ECAPA, or the learned null token."""
        if ecapa is None:
            return self.null_mem.reshape(1, D)
        return self.spk_proj_ecapa(ecapa.reshape(1, 192).astype(self.spk_proj_ecapa.weight.dtype))

    def cross_constants(self, mem):
        """Single memory token => cross-attn output = o(v(mem)), one (D,) vector per layer -> (LAYERS, D)."""
        out = []
        for blk in self.cross:
            v = blk.kv(mem.astype(blk.kv.weight.dtype))[..., D:]
            out.append(blk.o(v).reshape(D))
        return mx.stack(out)

    # --- one frame
    def _layer_full(self, blk, x, cross, keep_kv, last_only):
        B, T, _ = x.shape
        q, k, v = mx.split(blk.qkv(blk.n1(x)), 3, axis=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        if last_only:
            a = mx.fast.scaled_dot_product_attention(q[:, :, -1:], k, v, scale=self.scale)
            x = x[:, -1:]
        else:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        x = x + blk.o(a.transpose(0, 2, 1, 3).reshape(B, x.shape[1], D))
        x = x + blk.f2(nn.gelu(blk.f1(blk.n2(x))))
        return x + cross, ((k, v) if keep_kv else None)

    def _layer_step(self, blk, x, cross, kv):
        B = x.shape[0]
        q, k, v = mx.split(blk.qkv(blk.n1(x)), 3, axis=-1)
        k = mx.concatenate([kv[0], _heads(k)], axis=2)
        v = mx.concatenate([kv[1], _heads(v)], axis=2)
        a = mx.fast.scaled_dot_product_attention(_heads(q), k, v, scale=self.scale)
        x = x + blk.o(a.transpose(0, 2, 1, 3).reshape(B, 1, D))
        x = x + blk.f2(nn.gelu(blk.f1(blk.n2(x))))
        return x + cross, (k, v)

    def frame(self, h, cb0, cross, table, pick, n_q=N_Q):
        """Generate codebooks 1..n_q-1 for one frame.

        h (B, d_backbone) backbone hidden; cb0 (B,) int; cross (LAYERS, D); table from build_code_table();
        pick(level, logits (B, 2048) fp32) -> (B,) int chooses each code (sampling or teacher forcing).
        Returns (B, n_q) codes."""
        dt = self.in_proj.weight.dtype
        slots = [self.in_proj(h.astype(dt)) + self.level_emb.weight[0], table[0][cb0]]
        codes = [cb0]
        cross = [cross[i].astype(dt) for i in range(LAYERS)]
        kvs, prev = None, None
        for level in range(1, n_q):
            g = GROUP_OF_LEVEL[level]
            reused = level + 1 < n_q and GROUP_OF_LEVEL[level + 1] == g
            blocks = self.groups[g]
            if g == prev:                                    # shared band: one new position
                x = slots[level][:, None]
                new = []
                for li, blk in enumerate(blocks):
                    x, kv = self._layer_step(blk, x, cross[li], kvs[li])
                    new.append(kv)
                kvs = new
            else:                                            # new group: whole prefix, causal
                x = mx.stack(slots[: level + 1], axis=1)
                kvs = []
                for li, blk in enumerate(blocks):
                    x, kv = self._layer_full(blk, x, cross[li], keep_kv=reused, last_only=li == LAYERS - 1)
                    kvs.append(kv)
            logits = self.heads[level - 1](self.norm(x[:, -1])).astype(mx.float32)
            code = pick(level, logits)
            codes.append(code)
            if level + 1 < n_q:
                slots.append(table[level][code])
            prev = g
        return mx.stack(codes, axis=-1)


# ----------------------------------------------------------------------------- full model
class Maya2(nn.Module):
    def __init__(self, backbone_cfg: dict):
        super().__init__()
        from mlx_lm.models import qwen3

        cfg = dict(backbone_cfg)
        # transformers-5 configs keep rope_theta under rope_parameters; mlx-lm needs it at top level
        rp = cfg.get("rope_parameters") or {}
        cfg["rope_theta"] = float(cfg.get("rope_theta") or rp.get("rope_theta") or 1e6)
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope type {rp}")
        cfg.pop("rope_scaling", None)
        args = qwen3.ModelArgs.from_dict(cfg)
        self.lm = qwen3.Qwen3Model(args)
        self.n_layers = args.num_hidden_layers
        d = args.hidden_size
        self.cb0_head = nn.Linear(d, N_CODES + 1, bias=False)
        self.audio_in = AudioInputEmbed(d)
        self.speaker_anchor = SpeakerAnchor(d)
        self.depth = DepthDecoder(d)

    def embed_tokens(self, ids):
        return self.lm.embed_tokens(ids)

    def backbone(self, emb, cache):
        """emb (B, T, d) -> post-norm hidden (B, T, d)."""
        return self.lm(None, cache=cache, input_embeddings=emb)

    def cb0_logits(self, h):
        return self.cb0_head(h).astype(mx.float32)

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.n_layers)]


def _sanitize(weights):
    out = {}
    for k, v in weights.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("backbone.lm.model."):
            out["lm." + k[len("backbone.lm.model."):]] = v
        elif k == "backbone.lm.lm_head.weight":
            continue                                  # tied to embed_tokens; text logits are never needed
        elif k.startswith("backbone.cb0_head."):
            out[k[len("backbone."):]] = v
        else:
            out[k] = v
    return out


def load_model(model_dir, depth_dtype=mx.float16, backbone_dtype=mx.bfloat16, quantize_bits=None, group_size=64):
    """Build Maya2 from a release dir (model*.safetensors + backbone/config.json)."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "backbone" / "config.json").read_text())
    model = Maya2(cfg)
    weights = {}
    for f in sorted(model_dir.glob("model*.safetensors")):
        weights.update(mx.load(str(f)))
    weights = _sanitize(weights)
    cast = {}
    for k, v in weights.items():
        if k.startswith("depth."):
            cast[k] = v.astype(depth_dtype)
        elif k.startswith("speaker_anchor."):
            cast[k] = v.astype(mx.float32)
        else:
            cast[k] = v.astype(backbone_dtype)
    model.load_weights(list(cast.items()), strict=True)
    if quantize_bits:
        # backbone transformer layers + cb0 head + depth trunks/heads. Kept full precision: embedding tables, the
        # speaker paths, and depth.in_proj / depth.code_up (measured: quantizing those two alone raises the depth
        # NLL 3.90 -> 4.46, while quantizing every trunk and head leaves it at 3.90)
        keep = ("depth.spk_proj", "speaker_anchor", "depth.cross", "depth.in_proj", "depth.code_up")

        def pred(path, m):
            if not isinstance(m, nn.Linear) or path.startswith(keep):
                return False
            return m.weight.shape[-1] % group_size == 0
        nn.quantize(model, group_size=group_size, bits=quantize_bits, class_predicate=pred)
    model.eval()
    mx.eval(model.parameters())
    return model
