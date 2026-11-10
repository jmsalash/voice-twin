"""Voice assistant server: a fine-tuned Mini-Omni2 answering spoken questions over a WebSocket.

  python assistant/server.py            (settings from the environment, see Config)

WebSocket /ws, JSON messages:
  -> {"type": "hello", "turnstile": "<token>"}       first message (token only if TURNSTILE_SECRET is set)
  <- {"type": "ready", "maxTurns": 20, "sessionSeconds": 300}  or {"type": "busy"} / {"type": "denied"} then close
  -> {"type": "audio", "data": "<base64 PCM16LE, 16 kHz mono>"}   the question, in one or more chunks
  -> {"type": "end", "id": 7}                         the question is complete: answer it (every reply carries its id)
  <- {"type": "heard", "text": "...", "ms": 300}     with RAG: what the question was transcribed as (for captions)
  <- {"type": "queued", "position": 1}               only while every answer slot is busy
  <- {"type": "text", "text": "piece"}  /  {"type": "audio", "data": "<base64 PCM16LE, 24 kHz mono>"}
  <- {"type": "done", "rephrase": false, "stats": {...}}
  -> {"type": "stop"}                                 the visitor talks over the answer: stop it
  <- {"type": "error", "message": "..."}; {"type": "limit", "reason": "turns" | "time"} then close
GET /health -> {"ok": true, "answering": 0, "slots": 4, "sessions": 0, "maxSessions": 4, "stepMs": 40}
  (for an "assistant offline" state). Answers are generated together (assistant/batch.py): a new question joins at
  the next decoder step, so a visitor never waits for someone else's answer to finish while a slot is free.
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


@dataclass
class Config:
    model_dir: str = env("MODEL_DIR", "/models/assistant")
    whisper: str = env("WHISPER_PATH", "/opt/whisper/small.pt")
    device: str = env("DEVICE", "cuda")
    host: str = env("HOST", "0.0.0.0")
    port: int = env("PORT", 8765, int)
    allowed_origins: list = field(default_factory=lambda: [o.strip() for o in env("ALLOWED_ORIGINS", "").split(",") if o.strip()])
    turnstile_secret: str = env("TURNSTILE_SECRET", "")
    max_sessions: int = env("MAX_SESSIONS", 4, int)  # open conversations
    slots: int = env("SLOTS", 4, int)  # answers generated together (batched); more wait for a free slot
    session_seconds: int = env("SESSION_SECONDS", 300, int)
    max_turns: int = env("MAX_TURNS", 20, int)
    max_question_seconds: float = env("MAX_QUESTION_SECONDS", 20, float)
    agree_words: int = env("AGREE_WORDS", 6, int)
    log_questions: int = env("LOG_QUESTIONS", 0, int)
    min_speech_seconds: float = env("MIN_SPEECH_SECONDS", 0.3, float)  # less than this: the press is ignored
    ssl_cert: str = env("SSL_CERTFILE", "")  # serve HTTPS directly (testing on a LAN); behind a tunnel, leave empty
    ssl_key: str = env("SSL_KEYFILE", "")
    rag_k: int = env("RAG_K", 2, int)  # answers retrieved and put in front of the question (if the model has rag/)
    min_margin: float = env("MIN_MARGIN", 0.02, float)  # retriever confidence below this: ask to rephrase


def pcm16_b64(x):
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def b64_pcm16(s):
    return np.frombuffer(base64.b64decode(s), dtype="<i2").astype(np.float32) / 32768


def speech_seconds(wav, frame=320, floor=0.01):
    """Seconds of 20 ms frames louder than a quiet room (RMS above floor, or 4x the quietest frames)."""
    n = len(wav) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(1))
    thr = max(floor, 4 * float(np.percentile(rms, 10)))
    return float((rms > thr).sum()) * frame / 16000


def turnstile_ok(secret, token, ip):
    """Cloudflare Turnstile: checks the token the page's widget produced."""
    if not token:
        return False
    data = urllib.parse.urlencode({"secret": secret, "response": token, "remoteip": ip or ""}).encode()
    try:
        with urllib.request.urlopen("https://challenges.cloudflare.com/turnstile/v0/siteverify", data, timeout=10) as r:
            return bool(json.load(r).get("success"))
    except Exception:
        return False


def make_app(cfg, assistant):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from batch import AnswerSeq, Batcher
    from engine import CHUNK, SAMPLES_PER_FRAME

    app = FastAPI()
    from fastapi.middleware.cors import CORSMiddleware

    # pages on other origins may read /health (the WebSocket checks origins itself, see ALLOWED_ORIGINS)
    app.add_middleware(CORSMiddleware, allow_origins=cfg.allowed_origins or ["*"], allow_methods=["GET"], allow_headers=["*"])
    state = {"sessions": 0}
    batcher = Batcher(assistant, slots=cfg.slots)

    @app.get("/health")
    def health():
        return {"ok": True, "answering": batcher.busy, "slots": cfg.slots, "sessions": state["sessions"],
                "maxSessions": cfg.max_sessions, "stepMs": round(batcher.step_ms, 1)}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        origin = sock.headers.get("origin", "")
        if cfg.allowed_origins and origin not in cfg.allowed_origins:
            await sock.close(code=1008)
            return
        await sock.accept()
        if state["sessions"] >= cfg.max_sessions:
            await sock.send_json({"type": "busy"})
            await sock.close()
            return
        state["sessions"] += 1
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        started, turns, question = time.time(), 0, []
        last = {}  # this conversation's previous question and topic, for follow-ups
        try:
            hello = json.loads(await asyncio.wait_for(sock.receive_text(), 15))
            if cfg.turnstile_secret:
                ip = sock.headers.get("cf-connecting-ip") or (sock.client.host if sock.client else "")
                if not await asyncio.to_thread(turnstile_ok, cfg.turnstile_secret, hello.get("turnstile"), ip):
                    await sock.send_json({"type": "denied"})
                    await sock.close()
                    return
            await sock.send_json({"type": "ready", "maxTurns": cfg.max_turns, "sessionSeconds": cfg.session_seconds})
            if assistant.disclaimer_audio is not None:  # first person: who they are talking to, before anything else
                await sock.send_json({"type": "text", "id": "disclaimer", "text": assistant.disclaimer_text})
                step = CHUNK * SAMPLES_PER_FRAME
                for k in range(0, len(assistant.disclaimer_audio), step):
                    await sock.send_json({"type": "audio", "id": "disclaimer", "data": pcm16_b64(assistant.disclaimer_audio[k: k + step])})
                await sock.send_json({"type": "done", "id": "disclaimer", "rephrase": False, "stopped": False, "stats": {}})

            async def answer(wav, qid):
                """Joins the batch (several visitors' answers advance together on the GPU) and forwards the events."""
                out = asyncio.Queue()

                def emit(kind, v):  # called from the batch thread
                    loop.call_soon_threadsafe(out.put_nowait, (kind, v))
                    if kind == "done":
                        loop.call_soon_threadsafe(out.put_nowait, None)

                if speech_seconds(wav) < cfg.min_speech_seconds:  # pressed without speaking: ignore it
                    if cfg.log_questions:
                        print("   (no speech: ignored)", flush=True)
                    await sock.send_json({"type": "done", "id": qid, "empty": True, "rephrase": False, "stopped": False, "stats": {}})
                    return
                prep = None
                if assistant.rag is not None:  # transcribe + retrieve (in a thread: it uses the GPU too)
                    t0 = time.time()
                    try:
                        heard = await asyncio.to_thread(assistant.rag.transcribe, assistant.whisper, wav)
                        prep = await asyncio.to_thread(assistant.rag.prepare, heard, last.get("prev"))
                    except Exception as ex:  # never leave the visitor waiting in silence
                        print(f"   error before answering: {ex!r}", flush=True)
                        await sock.send_json({"type": "error", "id": qid, "message": "something went wrong"})
                        return
                    await sock.send_json({"type": "heard", "id": qid, "text": heard, "ms": int(1000 * (time.time() - t0))})
                    if not any(c.isalnum() for c in heard):  # nothing intelligible
                        await sock.send_json({"type": "done", "id": qid, "empty": True, "rephrase": False, "stopped": False, "stats": {}})
                        return
                    if cfg.log_questions:  # testing: what was heard and retrieved (visitors' questions, so off by default)
                        print(f"heard {heard!r} -> {prep['topics']} score {prep['scores'][0]:.3f} margin {prep['margin']:.3f}"
                              + (f" (follow-up: {prep['followup']})" if prep.get("followup") else ""), flush=True)
                    if prep["confident"]:  # remembered for a follow-up question ("and before that?")
                        last["prev"] = {"q": heard, "topic": prep["topics"][0]}
                if not batcher.free:  # every slot answering someone: this one starts when a slot frees up
                    await sock.send_json({"type": "queued", "id": qid, "position": batcher.pending.qsize() + 1})
                batcher.submit(AnswerSeq(assistant, wav, emit, stop.is_set, rag=prep))
                said = []
                while (item := await out.get()) is not None:
                    kind, v = item
                    if stop.is_set() and kind != "done":
                        continue  # the visitor left or talked over it: drain until the batch confirms
                    if kind == "text":
                        said.append(v)
                    try:
                        await send(kind, v, qid, "".join(said))
                    except Exception:  # the visitor disconnected mid-answer
                        stop.set()

            async def send(kind, v, qid, said=""):
                """Every reply carries the question's id, so the page can drop leftovers of an interrupted answer."""
                if kind == "text":
                    await sock.send_json({"type": "text", "id": qid, "text": v})
                elif kind == "audio":
                    await sock.send_json({"type": "audio", "id": qid, "data": pcm16_b64(v)})
                elif kind == "done":
                    if cfg.log_questions:
                        tag = ("(interrupted) " if v.get("stopped") else "(rephrase) " if v.get("rephrase")
                               else f"(rescued: {v.get('topic')}) " if v.get("rescued") else "")
                        print(f"   answer {tag}{said.strip()[:140]!r}", flush=True)
                        if (v.get("rephrase") or v.get("rescued")) and v.get("reason"):
                            print(f"      why: {v['reason']} | text row: {v.get('candidate', '')[:90]!r} | voice row: {v.get('own', '')[:90]!r}", flush=True)
                    await sock.send_json({"type": "done", "id": qid, "rephrase": bool(v.get("rephrase")), "stopped": bool(v.get("stopped")),
                                          "stats": {k: v[k] for k in ("firstAudioMs", "totalMs") if k in v}})
                elif kind == "error":
                    await sock.send_json({"type": "error", "id": qid, "message": v})

            current = None
            while True:
                if time.time() - started > cfg.session_seconds:
                    await sock.send_json({"type": "limit", "reason": "time"})
                    break
                msg = json.loads(await sock.receive_text())
                t = msg.get("type")
                if t == "audio":
                    question.append(b64_pcm16(msg["data"]))
                    if sum(len(q) for q in question) > cfg.max_question_seconds * 16000:
                        question = question[:-1]  # long enough: ignore the rest
                elif t == "end":
                    if current and not current.done():
                        stop.set()
                        await current
                    if not question:
                        continue
                    turns += 1
                    if turns > cfg.max_turns:
                        await sock.send_json({"type": "limit", "reason": "turns"})
                        break
                    wav, question = np.concatenate(question), []
                    stop.clear()
                    current = asyncio.create_task(answer(wav, msg.get("id")))
                elif t == "stop":
                    stop.set()
                    question = []
                else:
                    await sock.send_json({"type": "error", "message": f"unknown type {t}"})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except Exception as e:
            try:
                await sock.send_json({"type": "error", "message": f"{type(e).__name__}"})
            except Exception:
                pass
        finally:
            stop.set()
            state["sessions"] -= 1
            try:
                await sock.close()
            except Exception:
                pass

    web = os.environ.get("WEB_DIR", os.path.join(HERE, "web"))
    if os.path.isdir(web) and env("SERVE_TEST_PAGE", 1, int):  # the test page at / (set SERVE_TEST_PAGE=0 to turn off)
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def main():
    import uvicorn

    from engine import Assistant

    cfg = Config()
    assistant = Assistant(cfg.model_dir, cfg.whisper, cfg.device, cfg.agree_words, rag_k=cfg.rag_k, min_margin=cfg.min_margin)
    print(f"model loaded on {cfg.device} in {assistant.load_seconds} s; origins: {cfg.allowed_origins or 'any'}; "
          f"turnstile: {'on' if cfg.turnstile_secret else 'off'}; max sessions {cfg.max_sessions}, slots {cfg.slots}; "
          f"RAG: {f'on, {assistant.rag.n_index} wordings, k={cfg.rag_k}, min margin {cfg.min_margin}' if assistant.rag else 'off'}", flush=True)
    tls = {"ssl_certfile": cfg.ssl_cert, "ssl_keyfile": cfg.ssl_key} if cfg.ssl_cert and cfg.ssl_key else {}
    uvicorn.run(make_app(cfg, assistant), host=cfg.host, port=cfg.port, log_level="warning", **tls)


if __name__ == "__main__":
    main()
