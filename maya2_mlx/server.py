"""Local HTTP server (standard library only): a small web UI and an OpenAI-style speech endpoint.

    POST /v1/audio/speech   {"input", "voice", "language", "response_format": "wav"|"pcm", "seed", "normalize"}
                            wav: complete file; pcm: raw s16le 24 kHz mono, streamed as it is generated
    POST /v1/voices/clone   multipart: file=<audio>, name, text, language   -> registers a cloned voice
    GET  /v1/voices         built-in + cloned voices
    GET  /health
    GET  /                  web UI
No authentication: bind to 127.0.0.1 (the default) unless you add access control in front of it.
"""
from __future__ import annotations

import io
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .tokens import LANG_NAMES, LANGS, SR

MAX_BODY = 20 * 1024 * 1024


def _wav_bytes(wav):
    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, wav, SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _multipart(handler, body):
    from email.parser import BytesParser
    from email.policy import default

    msg = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + handler.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        out[name] = part.get_payload(decode=True) if part.get_filename() else part.get_content()
    return out


def make_handler(tts):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("request too large")
            return self.rfile.read(n)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ready"})
            if self.path == "/v1/voices":
                return self._json(200, {"voices": tts.voice_info(), "languages": LANG_NAMES})
            if self.path in ("/", "/index.html"):
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/v1/audio/speech":
                    return self._speech(json.loads(self._body() or b"{}"))
                if self.path == "/v1/voices/clone":
                    return self._clone(_multipart(self, self._body()))
                self._json(404, {"error": "not found"})
            except (ValueError, KeyError) as e:
                self._json(400, {"error": str(e)})
            except BrokenPipeError:
                pass

        def _speech(self, req):
            text = req.get("input") or ""
            lang = req.get("language") or req.get("lang") or "en"
            if lang not in LANGS:
                raise ValueError(f"language must be one of {LANGS}")
            voice = tts.get_voice(req.get("voice") or "Ahana")
            kw = dict(normalize=bool(req.get("normalize", True)), seed=req.get("seed"),
                      max_frames=req.get("max_frames"))
            fmt = req.get("response_format", "wav")
            with lock:
                if fmt == "pcm":
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/pcm")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for c in tts.stream(text, lang, voice, **kw):
                        b = (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes()
                        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    return
                wav, info = tts.generate(text, lang, voice, return_info=True, **kw)
            b = _wav_bytes(wav)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("X-Maya2-Info", json.dumps({k: info.get(k) for k in ("audio_s", "wall_s", "rtf", "stops")}))
            self.end_headers()
            self.wfile.write(b)

        def _clone(self, form):
            from .api import load_audio

            name = (form.get("name") or "").strip()
            if not name or not form.get("file"):
                raise ValueError("name and file are required")
            with tempfile.NamedTemporaryFile(suffix=".audio") as f:
                f.write(form["file"])
                f.flush()
                w = load_audio(f.name)
            with lock:
                v = tts.clone_voice(name, w, form.get("text", ""), form.get("language", "en"))
            self._json(200, {"voice": v.name, "mode": v.mode})

    return H


def serve(tts, host="127.0.0.1", port=8080):
    srv = ThreadingHTTPServer((host, port), make_handler(tts))
    print(f"[maya2-mlx] serving on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maya 2 · MLX</title>
<style>
:root{--bg:#f7f6f3;--fg:#1d1c1a;--mut:#6b6862;--card:#fff;--line:#e4e1da;--acc:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--fg:#ecebe8;--mut:#a3a09a;--card:#201f1d;--line:#34322f;--acc:#fb923c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
textarea{width:100%;min-height:120px;font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;align-items:center}
select,input[type=text],input[type=number]{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
button{font:inherit;font-weight:600;padding:8px 16px;border:0;border-radius:8px;background:var(--acc);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}label{color:var(--mut);font-size:13px}.stat{color:var(--mut);font-size:13px;margin-top:10px;min-height:1.5em}
audio{width:100%;margin-top:12px}h2{font-size:15px;margin:0 0 8px}
</style></head><body><main>
<h1>Maya 2 Native</h1><p class="sub">Text-to-speech in 12 languages, running locally with MLX.</p>
<div class="card">
<textarea id="t">नमस्ते! मैं माया हूँ। आज आपकी क्या मदद कर सकती हूँ?</textarea>
<div class="row"><label>Language <select id="lang"></select></label><label>Voice <select id="voice"></select></label>
<label>Seed <input id="seed" type="number" style="width:90px" placeholder="random"></label>
<label><input id="norm" type="checkbox" checked> normalise</label>
<button id="go">Speak</button></div>
<div class="stat" id="stat"></div><audio id="a" controls></audio></div>
<div class="card"><h2>Clone a voice</h2>
<div class="row"><input type="file" id="f" accept="audio/*"><input type="text" id="cn" placeholder="voice name">
<select id="cl"></select></div>
<div class="row" style="width:100%"><input type="text" id="ct" placeholder="exact transcript of the clip" style="flex:1"><button id="cb">Clone</button></div>
<div class="stat" id="cs">3–10 s of clean single-speaker audio. Only clone voices you have permission to use.</div></div>
</main><script>
const $=id=>document.getElementById(id);
async function load(){const r=await (await fetch('/v1/voices')).json();
 for(const s of [$('lang'),$('cl')]){s.innerHTML='';for(const [k,n] of Object.entries(r.languages)){const o=new Option(n,k);s.add(o)}}
 $('lang').value='hi';$('cl').value='en';const v=$('voice'),cur=v.value;v.innerHTML='';
 for(const x of r.voices){v.add(new Option(x.name+(x.description?' — '+x.description:''),x.name))}if(cur)v.value=cur}
$('go').onclick=async()=>{const b=$('go');b.disabled=true;$('stat').textContent='Generating…';const t0=performance.now();
 try{const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({input:$('t').value,language:$('lang').value,voice:$('voice').value,normalize:$('norm').checked,
  seed:$('seed').value?+$('seed').value:null})});
  if(!r.ok)throw new Error((await r.json()).error);const info=JSON.parse(r.headers.get('X-Maya2-Info')||'{}');
  $('a').src=URL.createObjectURL(await r.blob());$('a').play();
  $('stat').textContent=`${info.audio_s}s of audio in ${((performance.now()-t0)/1000).toFixed(1)}s (RTF ${info.rtf}) · stop: ${(info.stops||[]).join(', ')}`}
 catch(e){$('stat').textContent='Error: '+e.message}b.disabled=false};
$('cb').onclick=async()=>{const fd=new FormData();if(!$('f').files[0]){$('cs').textContent='Choose an audio file first.';return}
 fd.append('file',$('f').files[0]);fd.append('name',$('cn').value);fd.append('text',$('ct').value);fd.append('language',$('cl').value);
 $('cs').textContent='Cloning…';const r=await fetch('/v1/voices/clone',{method:'POST',body:fd});const j=await r.json();
 $('cs').textContent=r.ok?`Added voice "${j.voice}" (${j.mode}).`:'Error: '+j.error;if(r.ok){await load();$('voice').value=j.voice}};
load();
</script></body></html>"""
