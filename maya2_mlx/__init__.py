"""Maya 2 Native text-to-speech on Apple silicon with MLX."""
from .api import Maya2TTS, load_audio
from .generate import SamplingConfig
from .prompt import Voice
from .tokens import LANG_NAMES, LANGS, SR

__all__ = ["Maya2TTS", "SamplingConfig", "Voice", "LANGS", "LANG_NAMES", "SR", "load_audio"]
__version__ = "0.1.0"
