"""maya2 command line.

    maya2 say "नमस्ते, आप कैसे हैं?" --lang hi --voice Ahana -o hello.wav
    maya2 say "Hello there." --lang en --voice Siddhant --play          # stream to the speakers as it is generated
    maya2 voices
    maya2 clone me.wav --text "exact transcript" --lang en -o me.voice.npz
    maya2 say "வணக்கம்!" --lang ta --voice-file me.voice.npz -o ta.wav
    maya2 serve --port 8080                                             # web UI + OpenAI-style /v1/audio/speech
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from .tokens import LANG_NAMES, LANGS, SR


def _load(a):
    from . import Maya2TTS, SamplingConfig

    sc = SamplingConfig(seed=getattr(a, "seed", None))
    if getattr(a, "temperature", None):
        sc.cb0_temp = a.temperature
    return Maya2TTS.from_pretrained(a.model, quantize=None if a.bf16 else a.quantize, sampling=sc)


def _player():
    """Return push(chunk)/close() for live playback, or None if sounddevice is unavailable."""
    try:
        import sounddevice as sd
    except Exception:
        return None
    q = queue.Queue()
    buf = np.zeros(0, np.float32)
    done = threading.Event()

    def cb(out, frames, _t, _s):
        nonlocal buf
        while len(buf) < frames:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done.set()
                break
            buf = np.concatenate([buf, item])
        n = min(frames, len(buf))
        out[:n, 0] = buf[:n]
        out[n:, 0] = 0
        buf = buf[n:]

    stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", callback=cb, blocksize=1920)
    started = [False]
    pending = []

    def push(chunk):
        # pre-buffer ~0.5 s so a generator slower than real time stutters less
        pending.append(chunk)
        if not started[0] and sum(len(c) for c in pending) >= SR // 2:
            started[0] = True
            stream.start()
        if started[0]:
            while pending:
                q.put(pending.pop(0))

    def close():
        if not started[0]:
            stream.start()
        for c in pending:
            q.put(c)
        q.put(None)
        done.wait(timeout=600)
        time.sleep(0.2)
        stream.stop()
        stream.close()

    return push, close


def cmd_say(a):
    import soundfile as sf

    text = a.text if a.text != "-" else sys.stdin.read()
    if a.text_file:
        text = open(a.text_file, encoding="utf-8").read()
    tts = _load(a)
    voice = tts.load_voice(a.voice_file).name if a.voice_file else a.voice
    info = {}
    if a.play:
        pl = _player()
        if pl is None:
            print("(install `sounddevice` for live playback; playing the file afterwards instead)", file=sys.stderr)
    else:
        pl = None
    chunks = []
    for c in tts.stream(text, a.lang, voice, normalize=not a.no_normalize, seed=a.seed, max_frames=a.max_frames, info=info):
        chunks.append(c)
        if pl:
            pl[0](c)
    wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if pl:
        pl[1]()
    out = a.output or ("maya2_out.wav" if not a.play or pl is None else None)
    if out:
        sf.write(out, wav, SR)
    print(json.dumps({k: info.get(k) for k in ("audio_s", "wall_s", "rtf", "ttfa_s", "stops")}), file=sys.stderr)
    if out:
        print(out)
    if a.play and pl is None and out:
        subprocess.run(["afplay", out])


def cmd_voices(a):
    import json as _j
    from pathlib import Path

    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(a.model, allow_patterns=["voices/*"]))
    d = _j.loads((p / "voices" / "voices.json").read_text())
    print(f"{'voice':<11}{'gender':<8}{'age':<8}{'trained languages (>=1h)':<36}description")
    for v in d["voices"]:
        print(f"{v['name']:<11}{v.get('gender', ''):<8}{v.get('age_band', ''):<8}{' '.join(v.get('langs', [])):<36}"
              f"{v.get('description', '')}")
    print("\nlanguages: " + ", ".join(f"{k}={n}" for k, n in LANG_NAMES.items()))


def cmd_clone(a):
    tts = _load(a)
    v = tts.clone_voice(a.name or a.ref.rsplit("/", 1)[-1].split(".")[0], a.ref, a.text, a.lang, mode=a.mode,
                        save_to=a.output)
    print(f"saved voice '{v.name}' (mode {v.mode}) -> {a.output}")
    if a.mode is None and v.mode == "icl":
        print("note: speechbrain/torch not installed, so the clone uses reference audio only (mode icl). "
              "`pip install 'maya2-mlx[clone]'` enables ECAPA conditioning (icl_spk).", file=sys.stderr)


def cmd_serve(a):
    from .server import serve

    serve(_load(a), host=a.host, port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="maya2", description="Maya 2 Native TTS on Apple silicon (MLX)")
    ap.add_argument("--model", default="maya-research/maya2-native", help="HF repo id or local release dir")
    ap.add_argument("--quantize", type=int, choices=[4, 8], default=8, help="weight bits (default 8: lossless, faster)")
    ap.add_argument("--bf16", action="store_true", help="full-precision bf16/fp16 weights (slower, ~real time)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", help="synthesise speech")
    s.add_argument("text", nargs="?", default="", help="text ('-' = stdin)")
    s.add_argument("--text-file")
    s.add_argument("--lang", "-l", default="en", choices=LANGS)
    s.add_argument("--voice", "-v", default="Ahana")
    s.add_argument("--voice-file", help="cloned voice saved by `maya2 clone`")
    s.add_argument("--output", "-o")
    s.add_argument("--play", "-p", action="store_true", help="play while generating")
    s.add_argument("--seed", type=int)
    s.add_argument("--temperature", type=float, help="cb0 temperature (default 0.9)")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--no-normalize", action="store_true", help="skip number/date/currency normalisation")
    s.set_defaults(fn=cmd_say)

    v = sub.add_parser("voices", help="list built-in voices")
    v.set_defaults(fn=cmd_voices)

    c = sub.add_parser("clone", help="clone a voice from a 3-10 s reference clip")
    c.add_argument("ref", help="reference audio file")
    c.add_argument("--text", required=True, help="exact transcript of the reference clip")
    c.add_argument("--lang", required=True, choices=LANGS, help="language of the reference clip")
    c.add_argument("--name")
    c.add_argument("--mode", choices=["icl_spk", "icl"], default=None)
    c.add_argument("--output", "-o", required=True, help="where to save the voice (.npz)")
    c.set_defaults(fn=cmd_clone)

    sv = sub.add_parser("serve", help="web UI + HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    if a.cmd == "say" and not (a.text or a.text_file):
        ap.error("say needs text or --text-file")
    a.fn(a)


if __name__ == "__main__":
    main()
