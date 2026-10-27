"""The fast depth path (constant cross-attention, within-group KV reuse, last-position-only final layer) must equal
a literal port of DepthV2.generate: per-level full-prefix recompute with real cross-attention over the memory."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maya2_mlx.model import D, GROUP_OF_LEVEL, HEAD, LAYERS, DepthDecoder, _heads  # noqa: E402


def naive_level_logits(d, h, codes_prev, level, mem):
    """DepthV2._prefix + _run (reference semantics), one level."""
    parts = [d.in_proj(h)[:, None] + d.level_emb.weight[:1]]
    ks = mx.arange(level)
    e = d.code_up(d.code_lo(codes_prev[:, :level] + ks * 2048)) + d.level_emb(ks + 1)[None]
    x = mx.concatenate(parts + [e], axis=1)
    B, T, _ = x.shape
    for li in range(LAYERS):
        blk = d.groups[GROUP_OF_LEVEL[level]][li]
        q, k, v = mx.split(blk.qkv(blk.n1(x)), 3, axis=-1)
        a = mx.fast.scaled_dot_product_attention(_heads(q), _heads(k), _heads(v), scale=HEAD ** -0.5, mask="causal")
        x = x + blk.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        x = x + blk.f2(nn.gelu(blk.f1(blk.n2(x))))
        cb = d.cross[li]
        cq = _heads(cb.q(cb.n(x)))
        ck, cv = mx.split(cb.kv(mem), 2, axis=-1)
        a = mx.fast.scaled_dot_product_attention(cq, _heads(ck), _heads(cv), scale=HEAD ** -0.5)
        x = x + cb.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
    return d.heads[level - 1](d.norm(x[:, -1]))


def test_fast_equals_naive():
    mx.random.seed(0)
    d = DepthDecoder(2048)
    # give the random init realistic scales (LayerNorm weights 1, small linears)
    d.update(nn.utils.tree_map(lambda p: p * 0.5, d.parameters()))
    d.update({"null_mem": mx.random.normal((1, 1, D))})
    B = 2
    h = mx.random.normal((B, 2048))
    codes = mx.random.randint(0, 2048, (B, 32))
    for mem in (d.speaker_memory(None)[None], d.speaker_memory(mx.random.normal((192,)))[None]):
        mem = mx.broadcast_to(mem, (B, 1, D))
        cross = d.cross_constants(mem[:1, 0])
        table = d.build_code_table()
        got = []
        d.frame(h, codes[:, 0], cross, table, lambda lvl, lg: (got.append(lg), codes[:, lvl])[1])
        for level in (1, 2, 8, 9, 10, 16, 17, 24, 25, 31):
            ref = naive_level_logits(d, h, codes, level, mem)
            err = float(mx.abs(got[level - 1] - ref).max() / mx.abs(ref).max())
            assert err < 1e-4, f"level {level}: rel err {err}"
    print("depth fast path == naive DepthV2.generate: OK")


if __name__ == "__main__":
    test_fast_equals_naive()
