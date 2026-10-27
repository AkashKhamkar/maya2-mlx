"""Frame-by-frame generation: the official engine's sampler, stop rule and guards (maya3/engine.py), single stream.

Per frame (80 ms of audio):
  cb0   : logits / 0.9, + loop-guard <aeos> bias, presence penalty -0.8 on codes seen in the last 12 frames,
          top-64 over all 2049 classes, softmax. After a text-conditioned minimum length the piece stops when
          argmax == <aeos>, p(<aeos>) >= 0.35 or the cumulative stop hazard >= 0.6. Otherwise <aeos> is
          removed and cb0 is sampled.
  depth : codebooks 1..31 at temperature 0.7, top-64.
  back  : the 32-code frame is embedded and fed to the backbone for the next frame.
Guards: duration cap from a per-language prior, loop detector on the cb0 history (EOS boost, then backstop).
Long text is split into ~20 s pieces, each generated in a fresh context, joined with 150 ms of silence.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import mlx.core as mx
import numpy as np

from . import guards as G
from .tokens import AEOS_CLASS, FRAME, N_CODES, N_Q, SR

ASSETS = Path(__file__).parent / "assets"
FPS = 12.5
TERMINALS = "।.!?;।॥"
PRIOR_DEFAULT = {"a": 2.0, "b": 0.72, "resid_p99": 90.0}
REASONS = {2: "aeos_argmax", 3: "aeos_tau", 4: "aeos_hazard"}


@dataclasses.dataclass
class SamplingConfig:
    cb0_temp: float = 0.9
    cb0_top_k: int = 64
    rep_penalty: float = 0.8        # presence penalty on cb0 codes seen in the last rep_window frames
    rep_window: int = 12
    depth_temp: float = 0.7
    depth_top_k: int = 64
    tau: float = 0.35               # stop when p(<aeos>) >= tau
    hazard: float = 0.6             # stop when 1 - prod(1 - p_aeos) >= hazard
    hard_cap: int = 1500            # frames per piece (120 s)
    chunk_s: float = 20.0           # long text is split into pieces of about this many seconds (0 = off)
    gap_s: float = 0.15             # silence between pieces
    guards: bool = True
    seed: Optional[int] = None


# ----------------------------------------------------------------------------- text -> pieces, duration prior
def load_frames_prior(path=ASSETS / "text_frames_prior.json"):
    return {l: {"a": float(d["affine_a"]), "b": float(d["affine_b"]), "resid_p99": float(d["resid_p99"])}
            for l, d in json.loads(Path(path).read_text()).items()}


def expected_frames(prior, lang, nchar):
    p = prior.get(lang, PRIOR_DEFAULT)
    return p["a"] + p["b"] * nchar, p["resid_p99"]


def split_text(text, lang, prior, chunk_s):
    """Split at sentence terminals into pieces of about chunk_s seconds; commas / spaces for run-on text."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * FPS
    pieces = [p.strip() for p in re.split(f"(?<=[{re.escape(TERMINALS)}])\\s+", text) if p.strip()]
    out, cur = [], ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and expected_frames(prior, lang, len(cand))[0] > target:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    final = []
    for piece in out:
        if expected_frames(prior, lang, len(piece))[0] <= target * 1.3:
            final.append(piece)
            continue
        cur = ""
        for w in re.split(r"(?<=[,،])\s+|\s+", piece):
            cand = (cur + " " + w).strip() if cur else w
            if cur and expected_frames(prior, lang, len(cand))[0] > target:
                final.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final


# ----------------------------------------------------------------------------- sampling helpers
def _topk_mask(logits, k):
    kth = mx.min(mx.topk(logits, k, axis=-1), axis=-1, keepdims=True)
    return mx.where(logits < kth, -mx.inf, logits)


def _gumbel(shape):
    return -mx.log(-mx.log(mx.random.uniform(low=1e-20, high=1.0, shape=shape)))


class Generator:
    def __init__(self, model, sampling: SamplingConfig = None, compile_depth=True):
        self.model = model
        self.cfg = sampling or SamplingConfig()
        self.prior = load_frames_prior()
        self.dur = G.DurationPrior(str(ASSETS / "duration_prior.json"))
        self.gcfg = G.GuardConfig()
        self.table = model.depth.build_code_table()
        mx.eval(self.table)
        self._depth = mx.compile(self._depth_frame) if compile_depth else self._depth_frame

    # one frame of depth sampling: pure function of its inputs (noise passed in) so it can be compiled
    def _depth_frame(self, h, cb0, cross, noise):
        c = self.cfg

        def pick(level, logits):
            lg = _topk_mask(logits / c.depth_temp, c.depth_top_k)
            return mx.argmax(lg + noise[level - 1], axis=-1)

        codes = self.model.depth.frame(h, cb0, cross, self.table, pick)
        return codes, self.model.audio_in.frames(codes)

    def cross_for(self, voice):
        d = self.model.depth
        mem = d.speaker_memory(mx.array(voice.ecapa) if voice.uses_speaker else None)
        cross = d.cross_constants(mem)
        mx.eval(cross)
        return cross

    def plan(self, text, lang, max_frames=None):
        """Caps for one piece (engine._plan): (cap, loose_cap, minimum, loop-guard suspend windows)."""
        c, g = self.cfg, self.gcfg
        ef, er = expected_frames(self.prior, lang, len(text))
        loose = min(c.hard_cap, int(2.0 * (ef + er) + 40))
        cap = loose
        if c.guards:
            cap = min(cap, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        if max_frames:
            cap = min(cap, max(1, int(max_frames)))
        minimum = max(4, int(0.5 * ef))
        suspend = G.suspend_windows_v2(text, lang, self.dur) if c.guards else ()
        return cap, loose, minimum, suspend

    def frames(self, prompt_emb, cross, text, lang, max_frames=None, stats=None) -> Iterator[np.ndarray]:
        """Yield (32,) int code frames for one piece until the stop rule / a guard ends it."""
        m, c, g = self.model, self.cfg, self.gcfg
        cap, loose, minimum, suspend = self.plan(text, lang, max_frames)
        cache = m.make_cache()
        h = m.backbone(prompt_emb, cache)[:, -1]
        hist = np.full(g.hist, -1, np.int64)              # cb0 tail for the loop guard
        recent = []                                        # cb0 tail for the presence penalty
        hazard, aeos_bias, boost_at, n_boost = 0.0, 0.0, None, 0
        reason, n = "cap", 0
        t0 = time.perf_counter()
        while n < cap:
            adj = np.zeros(N_CODES + 1, np.float32)
            if c.rep_penalty and recent:
                adj[np.unique(recent)] -= c.rep_penalty
            adj[AEOS_CLASS] = aeos_bias
            logits = _topk_mask(m.cb0_logits(h)[0] / c.cb0_temp + mx.array(adj), c.cb0_top_k)
            probs = mx.softmax(logits, axis=-1)
            p_aeos, arg = probs[AEOS_CLASS], mx.argmax(probs)
            no_stop = logits.at[AEOS_CLASS].add(-mx.inf)
            cb0 = mx.argmax(no_stop + _gumbel(no_stop.shape))[None]
            # speculatively run the rest of the frame; a stop only discards it (one host sync per frame)
            codes, fb = self._depth(h, cb0, cross, _gumbel((N_Q - 1, 1, N_CODES)))
            h_next = m.backbone(fb[:, None], cache)[:, -1]
            mx.eval(p_aeos, arg, codes, h_next)
            if n >= minimum:
                pa = float(p_aeos)
                hazard = 1.0 - (1.0 - hazard) * (1.0 - pa)
                if int(arg) == AEOS_CLASS:
                    reason = "aeos_argmax"
                elif pa >= c.tau:
                    reason = "aeos_tau"
                elif hazard >= c.hazard:
                    reason = "aeos_hazard"
                if reason != "cap":
                    break
            frame = np.array(codes[0], np.int64)
            yield frame
            n += 1
            h = h_next
            k0 = int(frame[0])
            recent = (recent + [k0])[-c.rep_window:] if c.rep_window else []
            hist[:-1] = hist[1:]
            hist[-1] = k0
            if c.guards and g.loop and n >= g.start and n % g.every == 0 and \
                    not any(a <= n <= b for a, b in suspend):
                hit = G.loop_detect_batch(hist[None], np.array([n]), g)[0]
                if hit is not None:
                    if boost_at is None:
                        boost_at, aeos_bias, n_boost = n, g.boost, n_boost + 1
                    if n - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif boost_at is not None:
                    boost_at, aeos_bias = None, 0.0
        else:
            reason = "max_frames" if max_frames and n >= max_frames else ("cap_guard" if cap < loose else "cap")
        if stats is not None:
            dt = time.perf_counter() - t0
            stats.update({"reason": reason, "frames": n, "cap": cap, "loop_boosts": n_boost, "gen_s": dt,
                          "audio_s": n * FRAME / SR, "rtf": dt / max(1e-9, n * FRAME / SR)})
