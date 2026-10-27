# maya2-mlx

Run [Maya 2 Native](https://huggingface.co/maya-research/maya2-native) — Maya Research's streaming TTS for
**11 Indic languages + English** (20 built-in voices, zero-shot voice cloning) — natively on Apple silicon with
[MLX](https://github.com/ml-explore/mlx). No CUDA, no PyTorch at inference time.

The upstream release only runs on Linux + NVIDIA. This package is a from-scratch MLX port of its inference path.
It is verified by teacher-forced likelihood on a release sample, by Whisper transcripts of generated speech in
four languages and of a cloned voice, and by codec and depth-decoder equivalence tests (see below).

## Install

Requires an Apple silicon Mac and Python 3.10+. Tested with mlx 0.31.1, mlx-lm 0.31.3 and mlx-audio 0.6.2.

```bash
git clone https://github.com/AkashKhamkar/maya2-mlx && cd maya2-mlx
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[play]"          # [play] = live playback through your speakers (sounddevice)
```

The first run downloads the model (~5 GB, `maya-research/maya2-native`), the Mimi codec (~370 MB) and — when you
clone a voice — the ECAPA speaker encoder (~80 MB) into the Hugging Face cache.

## Samples

Generated on an M4 with the default 8-bit model, in [`examples/`](examples):

| File | Language · voice | Text |
|---|---|---|
| `hi_vaidehi.wav` | Hindi · Vaidehi | नमस्ते! आपका ₹2,500 का पेमेंट 15 तारीख को सफल हो गया है। |
| `mr_charvi_greeting.wav` | Marathi · Charvi | नमस्कार! आज तुमचा दिवस कसा गेला? मला आशा आहे की सगळं छान चाललं असेल. |
| `mr_anirban_bank.wav` | Marathi · Anirban | तुमच्या खात्यात ₹12,500 जमा झाले आहेत. पुढील हप्ता 5 तारखेला कापला जाईल. |
| `mr_malini_story.wav` | Marathi · Malini | एका छोट्या गावात एक म्हातारी आजी राहत होती… |
| `mr_siddhant_otp.wav` | Marathi · Siddhant | मला 98765 43210 या नंबरवर OTP आला नाही, कृपया पुन्हा पाठवा. |
| `ta_kayal.wav` | Tamil · Kayal | வணக்கம், எப்படி இருக்கிறீர்கள்? இன்று வானிலை மிகவும் நன்றாக இருக்கிறது. |
| `kn_satvik.wav` | Kannada · Satvik | ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ? |
| `en_vaidehi_story_stream.wav` | English · Vaidehi | a 34 s story, streamed in two pieces |
| `clone_siddhant_hindi.wav` | Hindi · voice cloned from an English reference | आज मौसम बहुत अच्छा है, चलिए बाहर घूमने चलते हैं। |
| `clone_ritvik_english.wav` | English · voice cloned from a Hindi reference | Hello! This is a cloned voice speaking English now. |

The cloned references are the release's own `samples/en_siddhant.wav` and `samples/hi_ritvik.wav`.

## Use it

### Command line

```bash
maya2 voices                                                     # list the 20 voices and their languages
maya2 say "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा।" --lang hi --voice Ahana -o hello.wav
maya2 say "Your order has shipped and will arrive tomorrow." -l en -v Siddhant --play
maya2 say --text-file story.txt -l te -v Nivedita -o story.wav   # long text is split into ~20 s pieces
maya2 --bf16 say "..." -l ta -v Kayal -o out.wav                 # full precision (default is lossless 8-bit)
```

Numbers, currency, dates, phone numbers, URLs… are expanded into words per language by Maya's own normaliser
(on by default, `--no-normalize` to skip). Inline tags such as `[laughs]`, `[sighs]`, `[whispers]` are best effort.

### Clone a voice

Use 3–10 s of clean single-speaker audio and its **exact** transcript. Only clone voices you have permission to use.

```bash
maya2 clone me.wav --text "exact words spoken in me.wav" --lang en -o me.voice.npz
maya2 say "வணக்கம், எப்படி இருக்கிறீர்கள்?" --lang ta --voice-file me.voice.npz -o me_tamil.wav
```

The reference language and output language can differ. The clone uses the reference audio in context plus an
ECAPA speaker embedding (`icl_spk`, the upstream default), computed with mlx-audio's MLX port of SpeechBrain's
ECAPA-TDNN.

### Web UI / HTTP API

```bash
maya2 serve --port 8080        # open http://127.0.0.1:8080
```

```bash
curl -s http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Your order has shipped.","language":"en","voice":"Ahana"}' -o speech.wav
# "response_format": "pcm" streams raw s16le 24 kHz mono while it is generated
```

The server has no authentication and binds to 127.0.0.1 by default.

### Python

```python
import soundfile as sf
from maya2_mlx import Maya2TTS

tts = Maya2TTS.from_pretrained()                     # 8-bit by default; quantize=None for bf16
wav = tts.generate("ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", lang="kn", voice="Satvik", seed=7)
sf.write("kn.wav", wav, 24000)

for chunk in tts.stream("Streaming works too.", lang="en", voice="Vaidehi"):
    ...                                               # float32 chunks of 80 ms as they are generated

tts.clone_voice("me", ref_audio="me.wav", ref_text="exact transcript", lang="hi", save_to="me.voice.npz")
wav = tts.generate("Hello from my cloned voice.", lang="en", voice="me")
print(tts.last_info)                                  # audio_s, wall_s, rtf, ttfa_s, stop reasons
```

## How the port works

| Component | Upstream (PyTorch/CUDA) | Here (MLX) |
|---|---|---|
| Backbone | Qwen3-1.7B, HF transformers + Triton kernels | `mlx_lm` Qwen3 fed with input embeddings, 8-bit (or bf16/4-bit) |
| cb0 head, audio-input embeddings, speaker anchor | `tts/model/*` | `maya2_mlx/model.py` |
| Depth decoder (codebooks 1–31) | `DepthV2` + CUDA graph | `DepthDecoder.frame`, fp16, compiled with `mx.compile` |
| Codec | `transformers.MimiModel` | `mlx_audio` Mimi (same weights, Moshi layout), streaming decode |
| Speaker encoder | SpeechBrain ECAPA (torch) | `mlx_audio` ECAPA (MLX), SpeechBrain fallback |
| Text normaliser, guards, priors | `maya3/text`, `maya3/guards.py` | vendored unchanged (pure Python) |

Per 80 ms frame: the backbone predicts codebook 0 (or end-of-audio); the depth decoder fills codebooks 1–31; the
32-code frame is embedded and fed back; Mimi turns it into 1920 samples. Sampling, stopping and guards follow the
official engine (`maya3/engine.py`): cb0 at temperature 0.9 / top-64 with a 0.8 presence penalty over 12 frames,
a controlled stop (argmax / p ≥ 0.35 / hazard ≥ 0.6 after a text-derived minimum length), depth at 0.7 / top-64,
a duration cap from a per-language prior, and a loop guard.

Two exact shortcuts from the upstream fast path are used in the depth decoder: with one speaker-memory token the
cross-attention reduces to a constant per layer, and levels that share a weight group reuse earlier positions' K/V.
`tests/test_depth.py` checks the fast path against a literal port of `DepthV2.generate`.

## Verification and speed

**Teacher-forced check** (`scripts/selfcheck.py`, model only): Maya's own release sample `samples/hi_ritvik.wav`
is encoded with Mimi and scored by the MLX model under its transcript and voice. A broken port lands near chance.

| Precision | cb0 NLL, right text + voice | wrong text | wrong voice | depth NLL (31 levels) |
|---|---:|---:|---:|---:|
| bf16 backbone / fp16 depth | **1.111** | 8.181 | 1.406 | **3.895** |
| 8-bit (default) | 1.121 | 8.131 | 1.401 | 3.900 |
| 4-bit | 1.354 | 8.403 | 1.413 | 4.346 |
| chance | 7.625 | | | 7.625 |

Two details were settled by this measurement rather than guessed. First, the frame-embedding RMSNorm needs
eps = 1.19e-7: with the bf16 eps (2⁻⁷) the cb0 NLL rises to 1.88. Second, the depth decoder's `in_proj` and
`code_up` must stay unquantized: quantizing them raises the depth NLL to 4.46.

**Intelligibility:** generated clips transcribed by Whisper large-v3-turbo:

| Clip | Whisper transcript |
|---|---|
| en · Siddhant | The EMI for your home loan is due by 12 October 2024. Please ensure your account has sufficient funds. |
| hi · Ahana | नमस्ते, आपका ओर्डर कल पहुंच जाएगा, क्या मैं आपकी और कोई मदद कर सकती हूँ? |
| ta · Kayal | வணக்கம் எப்படி இருக்கிறீர்கள் இன்று வானிலை மிகவும் நன்றாக இருக்கிறது. |

**Speed** on an Apple M4 (24 GB), mlx 0.31.1. One frame is 80 ms of audio; RTF below 1 means faster than real
time. Time to first audio was 0.12–0.25 s:

| Precision | ms / frame | RTF |
|---|---:|---:|
| bf16 / fp16 | ~77 | ~0.95 |
| 8-bit (default) | ~46 | ~0.58 |

Each frame reads the backbone weights once and a depth-decoder trunk 31 times (once per level), so speed appears
to be limited by memory bandwidth: halving the weight size (8-bit) gave a 1.67× speed-up, and compiling the depth
decoder made no difference. Only the M4 has been measured. Chips with more memory bandwidth (Pro/Max) should be
faster, but by how much is untested.

**Other checks that were run:**
- `tests/test_depth.py` checks the fast depth path against a literal port of `DepthV2.generate`.
- mlx-audio's Mimi matches `transformers.MimiModel`: cb0 codes 100% identical, all 32 levels 98.3%, decode
  correlation 0.999998, and streaming decode equal to offline decode.
- A voice cloned from the release's Siddhant sample scores ECAPA similarity 0.71 (Hindi output) and 0.78 (English
  output) to the reference, against 0.07 for an unrelated voice.

**Not yet run here:** `scripts/make_goldens.py` + `scripts/verify.py` compare every intermediate tensor (prompt ids,
embeddings, hidden states, cb0 and all 31 depth levels' logits) with the official PyTorch modules. They need a clone
of the upstream repo and torch. They were written for this port but have not been executed, so treat them as
untested.

## Licenses

Code in this repo: Apache-2.0. `maya2_mlx/text/`, `maya2_mlx/guards.py` and `maya2_mlx/assets/` are copied from
[MayaResearch/maya2-native](https://github.com/MayaResearch/maya2-native) (Apache-2.0, see
`LICENSE-maya2-native`). Model weights: Apache-2.0 (Maya Research). Upstream components keep their licenses:
Qwen3-1.7B-Base (Apache-2.0), Kyutai Mimi (CC-BY-4.0), SpeechBrain ECAPA-TDNN (Apache-2.0).

Maya 2 Native adds no audio watermark. As the model card asks: clone only voices you own or have explicit
permission to use, disclose synthetic speech to listeners, and do not use the model for impersonation, fraud,
harassment or to defeat voice authentication.
