"""Tests assistant/server.py: one visitor, or several at once (load test).

  python assistant/test_client.py --wavs q1.wav q2.wav [--url ws://127.0.0.1:8765/ws] [--visitors 3] [--out DIR]

Each visitor connects, asks every wav in turn (sent in 100 ms chunks, like a microphone), and plays nothing but
records: time to the first audio after the question ends, and whether the audio arrived faster than it plays
(it "keeps up" if, at every chunk, the audio received so far covers the time elapsed since the first chunk).
"""

import argparse
import asyncio
import base64
import json
import os
import time

import numpy as np
import soundfile as sf
import websockets


def b64(x):
    return base64.b64encode((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def load16(path):
    import librosa

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x


async def visitor(k, args, wavs, results):
    async with websockets.connect(args.url, max_size=2**26, origin=args.origin or None) as ws:
        await ws.send(json.dumps({"type": "hello"}))
        hello = json.loads(await ws.recv())
        if hello["type"] != "ready":
            results.append({"visitor": k, "refused": hello["type"]})
            return
        for qi, (name, x) in enumerate(wavs):
            for i in range(0, len(x), 1600):
                await ws.send(json.dumps({"type": "audio", "data": b64(x[i:i + 1600])}))
            t_end = time.time()
            await ws.send(json.dumps({"type": "end"}))
            first, got, keeps_up, text, queued, pcm = None, 0.0, True, "", None, []
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "queued":
                    queued = m["position"]
                elif m["type"] == "text":
                    text += m["text"]
                elif m["type"] == "audio":
                    a = np.frombuffer(base64.b64decode(m["data"]), "<i2")
                    pcm.append(a)
                    now = time.time()
                    first = first or now
                    if now - first > got + 0.05:  # playback would have run dry
                        keeps_up = False
                    got += len(a) / 24000
                elif m["type"] in ("done", "error", "limit"):
                    done = m
                    break
            r = {"visitor": k, "question": name, "queued": queued, "firstAudioS": round((first or time.time()) - t_end, 2),
                 "audioS": round(got, 1), "keepsUp": keeps_up, "rephrase": done.get("rephrase"), "text": text.strip()[:70]}
            results.append(r)
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                sf.write(os.path.join(args.out, f"v{k}_q{qi}.wav"), np.concatenate(pcm) if pcm else np.zeros(1, np.int16), 24000)
            print(r, flush=True)
            # a real visitor hears the answer out (playback started at the first chunk), then thinks
            await asyncio.sleep(args.pause + (max(0.0, (first or time.time()) + got - time.time()) if args.listen else 0))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--origin", default="")
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--visitors", type=int, default=1)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between a visitor's questions")
    ap.add_argument("--listen", action="store_true", help="like a real visitor: hear the whole answer before the pause")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wavs = [(os.path.basename(p), load16(p)) for p in args.wavs]
    results = []
    t = time.time()
    await asyncio.gather(*(visitor(k, args, wavs[k % len(wavs):] + wavs[:k % len(wavs)], results) for k in range(args.visitors)))
    ok = [r for r in results if "firstAudioS" in r]
    if ok:
        fa = sorted(r["firstAudioS"] for r in ok)
        print(f"\n{args.visitors} visitors, {len(ok)} answers in {time.time() - t:.0f} s: first audio median {fa[len(fa) // 2]} s, "
              f"worst {fa[-1]} s; keeps up with playback in {sum(r['keepsUp'] for r in ok)}/{len(ok)}; "
              f"refused {sum('refused' in r for r in results)}")


if __name__ == "__main__":
    asyncio.run(main())
