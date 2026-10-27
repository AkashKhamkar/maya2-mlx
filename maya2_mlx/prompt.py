"""Voices and prompt construction (maya3/prompt.py + tts/data/packing_v2.py upstream).

Span layout (one span = text then audio):
    [<|lang:A|>] text <|audio|> frame frame ... <aeos>

Prompts, ending at the target's <|audio|> sentinel (whose hidden state generates frame 0):
    named   : <|lang:B|> "{Name}: " + text <|audio|>                         speaker memory nulled
    plain   : <|lang:B|> text <|audio|>                                       speaker memory nulled
    spk     : same tokens as plain, ECAPA anchor + ECAPA depth memory
    icl     : <|lang:A|> ref <|audio|> codes <aeos> [<|lang:B|> if A != B] text <|audio|>   memory nulled
    icl_spk : icl + ECAPA (the default for a cloned voice)

The speaker anchor (ECAPA, or the learned null vector) is added at every span's <|audio|> sentinel.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

MODES = {"plain": (False, False), "spk": (False, True), "icl": (True, False), "icl_spk": (True, True),
         "named": (False, False)}          # (in-context reference audio, ECAPA speaker conditioning)
NAMED_TEMPLATE = "{{{name}}}: {text}"      # renders as "{Ahana}: text" -- the braces are literal
_CONTROL = re.compile(r"<\|[^|<>]{0,32}\|>")


@dataclasses.dataclass
class Voice:
    name: str
    mode: str = "plain"
    clips: list = dataclasses.field(default_factory=list)   # [{"text", "lang", "codes": (32, F) int}]
    ecapa: Optional[np.ndarray] = None                       # (192,) L2-normalised
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; choose one of {sorted(MODES)}")
        use_icl, use_spk = MODES[self.mode]
        if use_icl and not self.clips:
            raise ValueError(f"voice {self.name}: mode {self.mode} needs a reference clip")
        if use_spk and self.ecapa is None:
            raise ValueError(f"voice {self.name}: mode {self.mode} needs an ECAPA embedding")
        if self.ecapa is not None:
            self.ecapa = np.asarray(self.ecapa, np.float32).reshape(192)

    @property
    def uses_speaker(self):
        return MODES[self.mode][1]

    def save(self, path):
        """Persist a cloned voice (codes + ECAPA) so it can be reused without re-encoding."""
        path = Path(path)
        arrs = {f"codes_{i}": np.asarray(c["codes"], np.int32) for i, c in enumerate(self.clips)}
        if self.ecapa is not None:
            arrs["ecapa"] = self.ecapa
        meta = {"name": self.name, "mode": self.mode, "meta": self.meta,
                "clips": [{"text": c["text"], "lang": c["lang"]} for c in self.clips]}
        np.savez(path, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrs)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        clips = [{**c, "codes": d[f"codes_{i}"].astype(np.int64)} for i, c in enumerate(meta["clips"])]
        return cls(meta["name"], meta["mode"], clips=clips, ecapa=d["ecapa"] if "ecapa" in d else None,
                   meta=meta.get("meta", {}))


def load_named_voices(voices_json):
    d = json.loads(Path(voices_json).read_text())
    out = {}
    for it in d.get("voices", []):
        if it.get("mode", "named") != "named":
            continue                     # this release ships named voices only
        out[it["name"].lower()] = Voice(it["name"], "named", meta=it)
    return out


def clean_text(text):
    """Strip anything that would tokenize as a control token (<|audio|>, <|lang:xx|>, ...)."""
    return _CONTROL.sub(" ", text).strip()


def prompt_layout(tok, voice: Voice, text: str, lang: str):
    """-> list of ("ids", [int]) | ("frames", (F, 32) int) | ("aeos", None) pieces, ending at the target sentinel."""
    text = clean_text(text)
    if not text:
        raise ValueError("empty text")
    use_icl, _ = MODES[voice.mode]
    pieces = []
    if use_icl:
        for i, cl in enumerate(voice.clips):
            ids = ([tok.lang_ids[cl["lang"]]] if i == 0 else []) + tok.encode(clean_text(cl["text"])) + [tok.audio_id]
            pieces += [("ids", ids), ("frames", np.asarray(cl["codes"]).T), ("aeos", None)]
        xling = any(cl["lang"] != lang for cl in voice.clips)
        head = [tok.lang_ids[lang]] if xling else []
    else:
        head = [tok.lang_ids[lang]]
    body = NAMED_TEMPLATE.format(name=voice.name, text=text) if voice.mode == "named" else text
    pieces.append(("ids", head + tok.encode(body) + [tok.audio_id]))
    return pieces


def build_prompt(model, tok, voice: Voice, text: str, lang: str):
    """-> (prompt embeddings (1, T, d) in the backbone dtype, number of positions)."""
    pieces = prompt_layout(tok, voice, text, lang)
    anchor = model.speaker_anchor(mx.array(voice.ecapa) if voice.uses_speaker else None)
    dt = model.lm.embed_tokens.weight.dtype if hasattr(model.lm.embed_tokens, "weight") else mx.bfloat16
    parts = []
    for kind, val in pieces:
        if kind == "ids":
            e = model.embed_tokens(mx.array(val)).astype(mx.float32)
            # every span's <|audio|> sentinel is the last id of its text block
            e = mx.concatenate([e[:-1], e[-1:] + anchor[None]], axis=0)
            parts.append(e)
        elif kind == "frames":
            parts.append(model.audio_in.frames(mx.array(val.astype(np.int32))).astype(mx.float32))
        else:
            parts.append(model.audio_in.aeos()[None].astype(mx.float32))
    emb = mx.concatenate(parts, axis=0)[None].astype(dt)
    return emb, emb.shape[1]
