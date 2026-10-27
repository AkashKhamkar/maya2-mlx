"""Token contract of Maya 2 Native (tts/tokens.py upstream)."""
FRAME_HZ = 12.5
SR = 24000
FRAME = 1920               # samples per Mimi frame (80 ms)
N_Q = 32
N_CODES = 2048
AEOS_CLASS = N_CODES       # class 2048 of the cb0 head = <end-of-audio>

LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
LANG_NAMES = {"as": "Assamese", "bn": "Bengali", "en": "English", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
              "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"}

AUDIO = "<|audio|>"


def lang_token(lang):
    return f"<|lang:{lang}|>"


class Tokenizer:
    """The backbone's tokenizer.json (already carries the contract tokens), encoded without special tokens."""

    def __init__(self, backbone_dir):
        from tokenizers import Tokenizer as _T

        self.tok = _T.from_file(f"{backbone_dir}/tokenizer.json")
        self.audio_id = self._single(AUDIO)
        self.lang_ids = {l: self._single(lang_token(l)) for l in LANGS}

    def _single(self, t):
        i = self.tok.token_to_id(t)
        if i is None:
            raise ValueError(f"{t} missing from tokenizer.json")
        return i

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids
