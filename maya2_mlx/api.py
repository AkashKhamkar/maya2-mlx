"""Public API.

    from maya2_mlx import Maya2TTS
    tts = Maya2TTS.from_pretrained()                                   # maya-research/maya2-native
    wav = tts.generate("नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।", lang="hi", voice="Ahana")   # float32, 24 kHz
    for chunk in tts.stream("Your order has shipped.", lang="en", voice="Siddhant"): ...

    tts.clone_voice("me", ref_audio="me.wav", ref_text="<exact transcript>", lang="en")
    wav = tts.generate("வணக்கம்!", lang="ta", voice="me")
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx
import numpy as np

from .generate import Generator, SamplingConfig, split_text
from .model import load_model
from .prompt import MODES, Voice, build_prompt, load_named_voices
from .tokens import LANGS, SR, Tokenizer

DEFAULT_REPO = "maya-research/maya2-native"
RELEASE_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "maya3_config.json", "backbone/*", "voices/*"]


def _read_any(path):
    """Read an audio file. libsndfile covers wav/flac/mp3/ogg; anything else (m4a/aac from Voice Memos,
    webm/opus from browsers, ...) is decoded to 24 kHz mono wav with ffmpeg, or macOS afconvert."""
    import shutil
    import subprocess
    import tempfile

    import soundfile as sf

    try:
        return sf.read(str(path), dtype="float32")
    except sf.LibsndfileError:
        pass
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "decoded.wav")
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), out]
        elif shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(path), out]
        else:
            raise ValueError(f"cannot decode {path}: unsupported format (install ffmpeg, or use wav/flac/mp3)")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode or not os.path.exists(out):
            raise ValueError("cannot decode the audio file: unsupported format or corrupt file")
        return sf.read(out, dtype="float32")


def load_audio(path_or_array, sr=None):
    """path | (array, sr) | array at 24 kHz -> float32 mono 24 kHz."""
    if isinstance(path_or_array, (str, os.PathLike)):
        w, sr = _read_any(path_or_array)
    elif isinstance(path_or_array, tuple):
        w, sr = path_or_array
    else:
        w = path_or_array
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1) if w.shape[1] < w.shape[0] else w.mean(0)
    sr = sr or SR
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SR)
        w = resample_poly(w, SR // g, int(sr) // g).astype(np.float32)
    return w


class Maya2TTS:
    def __init__(self, model, tok, codec, voices, sampling=None, compile_depth=True):
        self.model, self.tok, self.codec = model, tok, codec
        self.gen = Generator(model, sampling, compile_depth=compile_depth)
        self._voices = voices                      # lower-case name -> Voice
        self._cross = {}
        self._ecapa = None
        self.lock = threading.Lock()               # one generation at a time (single Metal stream)
        self.last_info = {}

    @classmethod
    def from_pretrained(cls, repo_or_dir=DEFAULT_REPO, quantize: Optional[int] = 8, depth_dtype="fp16",
                        sampling: SamplingConfig = None, compile_depth=True, verbose=True):
        """quantize: 8 (default; measured lossless: cb0/depth NLL within 0.01, same ASR transcripts, ~1.65x faster),
        4 (smaller/faster, measurably worse), or None for bf16 weights."""
        t0 = time.perf_counter()
        p = Path(os.path.expanduser(str(repo_or_dir)))
        if not p.is_dir():
            from huggingface_hub import snapshot_download

            p = Path(snapshot_download(str(repo_or_dir), allow_patterns=RELEASE_PATTERNS))
        dd = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[depth_dtype]
        model = load_model(p, depth_dtype=dd, quantize_bits=quantize)
        tok = Tokenizer(p / "backbone")
        from .codec import Codec

        codec = Codec()
        voices = load_named_voices(p / "voices" / "voices.json") if (p / "voices" / "voices.json").is_file() else {}
        tts = cls(model, tok, codec, voices, sampling, compile_depth)
        if verbose:
            q = f"{quantize}-bit" if quantize else "bf16"
            print(f"[maya2-mlx] loaded {p.name} ({q} backbone, {depth_dtype} depth) in {time.perf_counter() - t0:.1f}s")
        return tts

    # ------------------------------------------------------------------ voices
    @property
    def voices(self):
        return [v.name for v in self._voices.values()]

    def voice_info(self):
        return [{"name": v.name, "mode": v.mode, **{k: v.meta.get(k) for k in ("gender", "age_band", "description", "langs")}}
                for v in self._voices.values()]

    def get_voice(self, name) -> Voice:
        if isinstance(name, Voice):
            return name
        if name is None:
            name = "Ahana"
        v = self._voices.get(str(name).lower())
        if v is None:
            raise KeyError(f"unknown voice {name!r}; available: {', '.join(self.voices)}")
        return v

    def add_voice(self, voice: Voice):
        self._voices[voice.name.lower()] = voice
        self._cross.pop(voice.name.lower(), None)
        return voice

    def clone_voice(self, name, ref_audio, ref_text, lang, mode: Optional[str] = None, save_to=None) -> Voice:
        """Zero-shot clone from a 3-10 s single-speaker clip and its exact transcript.

        mode: 'icl_spk' (reference audio + ECAPA, best; needs the `clone` extra: torch + speechbrain) or
        'icl' (reference audio only). Default: icl_spk when speechbrain is installed, else icl."""
        from . import speaker

        if lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}")
        if mode is None:
            mode = "icl_spk" if speaker.available() else "icl"
        if mode not in ("icl", "icl_spk", "spk"):
            raise ValueError("clone mode must be icl, icl_spk or spk")
        w = load_audio(ref_audio)
        if len(w) > 30 * SR:
            raise ValueError("reference audio longer than 30 s; 3-10 s works best")
        ecapa = None
        if MODES[mode][1]:
            if self._ecapa is None:
                self._ecapa = speaker.Ecapa()
            ecapa = self._ecapa(w)
        clips = []
        if MODES[mode][0]:
            if not ref_text:
                raise ValueError("ref_text (exact transcript of the reference clip) is required")
            clips = [{"text": ref_text, "lang": lang, "codes": self.codec.encode(w)}]
        v = Voice(name, mode, clips=clips, ecapa=ecapa, meta={"cloned": True, "ref_lang": lang})
        if save_to:
            v.save(save_to)
        return self.add_voice(v)

    def load_voice(self, path) -> Voice:
        return self.add_voice(Voice.load(path))

    def _cross_for(self, voice):
        k = voice.name.lower()
        if k not in self._cross:
            self._cross[k] = self.gen.cross_for(voice)
        return self._cross[k]

    # ------------------------------------------------------------------ synthesis
    def _prepare(self, text, lang, normalize):
        if lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if normalize:
            from .text import normalise_text

            text = normalise_text(text, lang)
        return split_text(text.strip(), lang, self.gen.prior, self.gen.cfg.chunk_s)

    def stream_codes(self, text, lang="en", voice=None, normalize=True, seed=None, max_frames=None,
                     info: Optional[dict] = None) -> Iterator[Optional[np.ndarray]]:
        """Yield (32,) code frames; None marks a piece boundary (insert gap_s of silence)."""
        v = self.get_voice(voice)
        pieces = self._prepare(text, lang, normalize)
        cross = self._cross_for(v)
        seed = self.gen.cfg.seed if seed is None else seed
        if seed is not None:
            mx.random.seed(int(seed))
        stops, total, t0, ttfa = [], 0, time.perf_counter(), None
        for i, piece in enumerate(pieces):
            if i:
                yield None
            emb, _ = build_prompt(self.model, self.tok, v, piece, lang)
            st = {}
            left = None if max_frames is None else max_frames - total
            if left is not None and left <= 0:
                break
            for fr in self.gen.frames(emb, cross, piece, lang, max_frames=left, stats=st):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total += 1
                yield fr
            stops.append(st)
        wall = time.perf_counter() - t0
        audio_s = total * 0.08 + self.gen.cfg.gap_s * max(0, len(stops) - 1)
        res = {"pieces": len(pieces), "frames": total, "audio_s": round(audio_s, 2), "wall_s": round(wall, 2),
               "rtf": round(wall / max(audio_s, 1e-9), 3), "ttfa_s": None if ttfa is None else round(ttfa, 3),
               "stops": [s.get("reason") for s in stops], "text": pieces}
        self.last_info = res
        if info is not None:
            info.update(res)

    def stream(self, text, lang="en", voice=None, normalize=True, seed=None, max_frames=None,
               info: Optional[dict] = None) -> Iterator[np.ndarray]:
        """Yield float32 24 kHz audio chunks (80 ms each) as they are generated."""
        gap = np.zeros(int(self.gen.cfg.gap_s * SR), np.float32)
        with self.lock, self.codec.stream() as step:
            for fr in self.stream_codes(text, lang, voice, normalize, seed, max_frames, info):
                yield gap if fr is None else step(fr)

    def generate(self, text, lang="en", voice=None, normalize=True, seed=None, max_frames=None,
                 return_info=False):
        """Synthesise text -> float32 waveform at 24 kHz (decoded in one pass per piece)."""
        info = {}
        out, cur = [], []
        gap = np.zeros(int(self.gen.cfg.gap_s * SR), np.float32)
        with self.lock:
            for fr in self.stream_codes(text, lang, voice, normalize, seed, max_frames, info):
                if fr is None:
                    out += [self.codec.decode(np.stack(cur)) if cur else np.zeros(0, np.float32), gap]
                    cur = []
                else:
                    cur.append(fr)
            if cur:
                out.append(self.codec.decode(np.stack(cur)))
        wav = np.concatenate(out) if out else np.zeros(0, np.float32)
        return (wav, info) if return_info else wav
