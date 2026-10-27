"""Kyutai Mimi (32 codebooks, 12.5 Hz, 24 kHz) via mlx-audio."""
from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"   # same weights as kyutai/mimi, original layout


class Codec:
    def __init__(self, repo=MIMI_REPO, filename=MIMI_FILE):
        from mlx_audio.codec.models.mimi.mimi import Mimi

        self.mimi = Mimi.from_pretrained(repo, filename)
        self.lock = threading.Lock()       # the mlx-audio module keeps its streaming state on itself

    def encode(self, wav24: np.ndarray) -> np.ndarray:
        """float32 mono 24 kHz -> (32, F) int64 codes."""
        with self.lock:
            codes = self.mimi.encode(mx.array(np.asarray(wav24, np.float32))[None, None])
            mx.eval(codes)
            self.mimi.reset_state()
        return np.array(codes[0], np.int64)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """(F, 32) codes -> float32 waveform."""
        if len(codes) == 0:
            return np.zeros(0, np.float32)
        with self.lock:
            pcm = self.mimi.decode(mx.array(np.asarray(codes, np.int32).T)[None])
            mx.eval(pcm)
            self.mimi.reset_state()
        return np.clip(np.array(pcm[0, 0], np.float32), -1.0, 1.0)     # the engine clamps PCM too

    def stream(self):
        """Context manager yielding a function frame (32,) -> 1920 samples; holds the codec for its duration."""
        return _Stream(self)


class _Stream:
    def __init__(self, codec):
        self.codec = codec

    def __enter__(self):
        from mlx_audio.codec.models.mimi.mimi import MimiStreamingDecoder

        self.codec.lock.acquire()
        self.dec = MimiStreamingDecoder(self.codec.mimi)
        return self.step

    def step(self, frame):
        pcm = self.codec.mimi.decode_step(mx.array(np.asarray(frame, np.int32))[None, :, None])
        return np.clip(np.array(pcm[0, 0], np.float32), -1.0, 1.0)

    def __exit__(self, *exc):
        self.codec.mimi.reset_state()
        self.codec.lock.release()
