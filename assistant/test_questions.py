"""Asks the running server a list of spoken questions and records the answers (one visitor, in turn).

  python assistant/test_questions.py questions.json out.jsonl [--url ws://127.0.0.1:8765/ws]
  questions.json: [{"wav": path, "question": text, "id": topic, "kind": "cv" | "offtopic", "answer": expected}]
"""

import argparse
import asyncio
import json
import time

import websockets

from test_client import b64, load16


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    args = ap.parse_args()
    qs = json.load(open(args.questions))
    open(args.out, "w").close()
    res = []
    async with websockets.connect(args.url, max_size=2**26, **({"ssl": __import__("ssl")._create_unverified_context()} if args.url.startswith("wss") else {})) as ws:
        await ws.send(json.dumps({"type": "hello"}))
        assert json.loads(await ws.recv())["type"] == "ready"
        await asyncio.sleep(0.3)
        while True:  # skip the disclaimer (first person), if any
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
            except asyncio.TimeoutError:
                break
            if m.get("type") == "done":
                break
        for q in qs:
            x = load16(q["wav"])
            for i in range(0, len(x), 1600):
                await ws.send(json.dumps({"type": "audio", "data": b64(x[i:i + 1600])}))
            t = time.time()
            await ws.send(json.dumps({"type": "end", "id": len(res) + 1}))
            text, heard, first, audio, done = "", None, None, 0.0, None
            while True:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 60))
                except asyncio.TimeoutError:  # a stuck answer must not stall the whole test
                    done = {"type": "timeout"}
                    break
                if m["type"] == "heard":
                    heard = m["text"]
                elif m["type"] == "text":
                    text += m["text"]
                elif m["type"] == "audio":
                    first = first or time.time() - t
                    audio += len(m["data"]) * 3 / 4 / 2 / 24000
                elif m["type"] in ("done", "error", "limit"):
                    done = m
                    break
            r = {**q, "heard": heard, "model": text.strip(), "rephrase": done.get("rephrase"), "end": done["type"],
                 "firstAudioS": round(first or 0, 2), "audioS": round(audio, 1)}
            res.append(r)
            with open(args.out, "a") as f:  # saved as it goes
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r['firstAudioS']:5.2f}s {'REPHRASE ' if r['rephrase'] else ''}{q['question'][:45]!r} -> {r['model'][:70]!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
