# Connecting a website to the voice assistant

Everything a web page needs to talk to the assistant server (`assistant/server.py`): the endpoint, the messages,
the audio formats, a hands-free (always listening) client, and what the page must show. The server does the
listening, the answering and the speaking; the page only captures the microphone, plays audio and shows text.

## 1. Endpoints

| | |
|---|---|
| WebSocket | `wss://<host>/ws` (the conversation) |
| Health | `GET https://<host>/health` -> `{"ok": true, "answering": 0, "slots": 4, "sessions": 0, "maxSessions": 4, "stepMs": 40}` |

`<host>` is wherever the server is reachable over HTTPS, e.g. a Cloudflare Tunnel hostname such as
`assistant.example.com` pointing at the server's port 8765. Browsers only allow the microphone on HTTPS pages, and an
HTTPS page can only open `wss://` sockets.

Use `/health` to decide whether to show the assistant at all: if it doesn't answer within ~3 s, show "the assistant is
offline" instead of a broken widget. `sessions >= maxSessions` means it is busy right now.

## 2. Server settings that concern the page

| setting | what to set |
|---|---|
| `ALLOWED_ORIGINS` | your site's origin(s), e.g. `https://example.com,https://www.example.com`. Other origins are refused. Empty = any (testing only). |
| `TURNSTILE_SECRET` | optional: if set, the page must send a Cloudflare Turnstile token in `hello` (see 3). |
| `MAX_SESSIONS`, `SLOTS` | open conversations / answers generated together (default 4 / 4). |
| `SESSION_SECONDS`, `MAX_TURNS` | per conversation (default 300 s / 20 questions), then the server sends `limit` and closes. |

## 3. The conversation

All messages are JSON text frames.

```
page                                   server
 | -- {"type":"hello","turnstile":T} --> |   first message (turnstile only if the server requires it)
 | <-- {"type":"ready", ...} ----------- |   or {"type":"busy"} / {"type":"denied"}, then the socket closes
 | <-- disclaimer: text, audio..., done  |   first-person assistants say who they are first (id "disclaimer")
 |                                       |
 | -- {"type":"audio","data":B64} -----> |   the question: 16 kHz mono PCM16, one or more chunks
 | -- {"type":"end","id":7} -----------> |   the question is complete: answer it
 | <-- {"type":"heard","id":7,...} ----- |   what was understood (show it as a caption)
 | <-- {"type":"text","id":7,...} ------ |   the answer's text, in pieces
 | <-- {"type":"audio","id":7,...} ----- |   the answer's audio, 24 kHz mono PCM16, in ~0.3 s chunks: play in order
 | <-- {"type":"done","id":7,...} ------ |   end of the answer
 |                                       |
 | -- {"type":"stop"} -----------------> |   the visitor talks over the answer: stop it
```

Details:

* **ids**: the page picks an id for each question (any number or string) and every reply carries it. After an
  interruption, drop any message whose id is not the current question's: pieces of the old answer may still arrive.
* **`heard`**: `{"type":"heard","id":7,"text":"What did you do in Berlin?","ms":310}`.
* **`done`**: `{"type":"done","id":7,"rephrase":false,"stopped":false,"empty":false,"stats":{"firstAudioMs":850}}`.
  `rephrase: true` means the assistant asked the visitor to rephrase (it wasn't sure: that is normal and safe).
  `empty: true` means no speech was found in what was sent (nothing is said back).
* **`queued`**: `{"type":"queued","id":7,"position":1}` if every answer slot is busy; the answer follows when one frees.
* **`limit`**: `{"type":"limit","reason":"time"|"turns"}`, then the socket closes. Offer to start again.
* **`error`**: `{"type":"error","id":7,"message":"..."}`: show a short "something went wrong", keep the session.
* Audio is base64 of little-endian 16-bit PCM. Send at most ~20 s per question (longer is cut).

## 4. Hands-free: always listening

The server needs whole questions. In hands-free mode the page decides where a question ends with a voice activity
detector (VAD) running in the browser; the protocol is the same as with a push-to-talk button.

* **Microphone constraints**: `{echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1}`.
  Echo cancellation removes the assistant's own voice coming out of the speakers, so it does not hear itself;
  noise suppression removes steady background noise.
* **VAD**: [`@ricky0123/vad-web`](https://github.com/ricky0123/vad) (Silero VAD in the browser, ISC licence) gives one
  callback when speech starts and one with the whole utterance (Float32Array, 16 kHz) when it ends.
* **Interrupting**: when the VAD hears speech start while the assistant is talking, stop playback at once, send
  `stop`, and drop the rest of that answer (ids).
* **Turn-taking**: wait ~0.6-0.8 s of silence before ending a question (the VAD's "redemption" setting), so a pause in
  the middle of a sentence doesn't cut it.
* **Headphones** are not needed, but on laptops with loud speakers, echo cancellation is not perfect: ignore speech
  that starts in the first ~0.3 s of the assistant's audio, or require a slightly higher VAD threshold while it talks.

### Minimal client

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js"></script>
<script>
const HOST = "assistant.example.com";            // your assistant's HTTPS host
let ws, out, next = 0, sources = [], qid = 0, current = "disclaimer";

function b64(f32) {                               // Float32 [-1, 1] -> base64 PCM16
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
  let s = "", b = new Uint8Array(pcm.buffer);
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
  return btoa(s);
}
function play(data) {                             // base64 PCM16 24 kHz, played back to back
  const pcm = new Int16Array(Uint8Array.from(atob(data), c => c.charCodeAt(0)).buffer);
  const buf = out.createBuffer(1, pcm.length, 24000), ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = out.createBufferSource(); src.buffer = buf; src.connect(out.destination);
  next = Math.max(next, out.currentTime + 0.05); src.start(next); next += buf.duration;
  sources.push(src); src.onended = () => sources = sources.filter(x => x !== src);
}
function stopPlayback() { sources.forEach(s => { try { s.stop(); } catch {} }); sources = []; next = 0; }

async function start() {                          // call from a click ("Start conversation"): browsers need one
  out = new AudioContext({ sampleRate: 24000 }); await out.resume();
  ws = new WebSocket(`wss://${HOST}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "hello" }));
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id !== undefined && m.id !== current) return;          // leftovers of an interrupted answer
    if (m.type === "audio") play(m.data);
    else if (m.type === "heard") showCaption("You: " + m.text);
    else if (m.type === "text") appendAnswer(m.text);
    else if (m.type === "busy") showStatus("The assistant is busy. Try again in a minute.");
    else if (m.type === "limit") showStatus("Conversation ended. Start again?");
  };
  const mic = await vad.MicVAD.new({            // `vad` is the library's global
    additionalAudioConstraints: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    redemptionFrames: 8,                         // ~0.8 s of silence ends the question
    onSpeechStart: () => {                        // the visitor talks over the answer
      if (sources.length) { ws.send(JSON.stringify({ type: "stop" })); stopPlayback(); current = null; }
    },
    onSpeechEnd: (audio) => {                     // Float32Array, 16 kHz: one whole question
      current = ++qid;
      ws.send(JSON.stringify({ type: "audio", data: b64(audio) }));
      ws.send(JSON.stringify({ type: "end", id: current }));
    },
  });
  mic.start();
}
</script>
```

(`showCaption`, `appendAnswer`, `showStatus` are the page's own UI. The library versions in the two `<script>` tags are
examples: check the current ones in the vad-web README and pin the versions you test with.)

## 5. What the page must show

* **A visible label** while the assistant is available: e.g. *"AI voice. Not a live person, and it can make mistakes."*
  For a first-person "digital twin" this is required, together with the spoken disclaimer the server plays first.
* **A start control.** Browsers only allow sound and the microphone after a click, so hands-free starts with one
  "Start conversation" click, which also asks for microphone permission. Offer a way to stop listening.
* **Captions**: what was heard and what is answered (`heard` and `text`). They help when audio fails and make it
  accessible.
* **Status**: connecting, listening, thinking, offline, busy.

## 6. Testing against a local server

A server on a LAN with a self-signed certificate (`SSL_CERTFILE`/`SSL_KEYFILE`) works for testing: first open
`https://<server-ip>:8765/health` in the browser and accept the certificate warning, otherwise the page's `wss://`
connection fails silently. The server also serves a push-to-talk test page at `https://<server-ip>:8765/`.
