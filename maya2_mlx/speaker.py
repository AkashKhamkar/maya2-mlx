"""ECAPA-TDNN speaker embedding (speechbrain/spkrec-ecapa-voxceleb), the recipe of maya3/api.py:
24 kHz -> 16 kHz, 80-bin fbank, sentence mean normalisation, ECAPA-192, L2-normalised.

Preferred: mlx-audio's MLX port of the same checkpoint (parity-tested against SpeechBrain upstream).
Fallback: SpeechBrain on PyTorch CPU (`pip install torch torchaudio speechbrain`).
Without either, voices can still be cloned in `icl` mode (reference audio + transcript only).
"""
from __future__ import annotations

import numpy as np

MLX_ECAPA = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
MLX_ECAPA_REV = "e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77"
ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"


def _mlx_extractor_cls():
    try:
        from mlx_audio.codec.models.ecapa_tdnn.speaker import EcapaSpeakerEmbeddingExtractor
        return EcapaSpeakerEmbeddingExtractor
    except Exception:
        return None


def _speechbrain_ok():
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def available():
    return _mlx_extractor_cls() is not None or _speechbrain_ok()


def _to16k(wav24):
    from scipy.signal import resample_poly

    if wav24 is None or len(wav24) < 2400:
        raise ValueError("reference audio is too short (need >= 0.1 s; 3-10 s recommended)")
    return resample_poly(np.asarray(wav24, np.float32), 2, 3).astype(np.float32)


class Ecapa:
    def __init__(self):
        cls = _mlx_extractor_cls()
        if cls is not None:
            self.backend = "mlx"
            self.ex = cls.from_pretrained(MLX_ECAPA, revision=MLX_ECAPA_REV)
        elif _speechbrain_ok():
            self.backend = "speechbrain"
            self._init_speechbrain()
        else:
            raise RuntimeError("no ECAPA backend: update mlx-audio, or install torch + speechbrain")

    def _init_speechbrain(self):
        import torch
        from huggingface_hub import hf_hub_download
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization

        self.torch = torch
        self.fbank = Fbank(n_mels=80).eval()
        self.norm = InputNormalization(norm_type="sentence", std_norm=False).eval()
        self.emb = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                              dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192).eval()
        ckpt = hf_hub_download(ECAPA_REPO, "embedding_model.ckpt")
        self.emb.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)

    def __call__(self, wav24: np.ndarray) -> np.ndarray:
        """float32 mono 24 kHz -> (192,) L2-normalised embedding."""
        w16 = _to16k(wav24)
        if self.backend == "mlx":
            return np.array(self.ex.encode(w16), np.float32)
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(w16)[None]
            L = torch.ones(1)
            e = self.emb(self.norm(self.fbank(x), L), L).reshape(-1)
            return torch.nn.functional.normalize(e, dim=0).numpy().astype(np.float32)
