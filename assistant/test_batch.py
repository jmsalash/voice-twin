"""Batched answers must match answers generated alone; measures latency with 1, 2 and 4 questions at once.

  python assistant/test_batch.py MODEL_DIR WHISPER_PATH q1.wav q2.wav ...
"""

import sys
import threading
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from batch import AnswerSeq, Batcher  # noqa: E402
from engine import Assistant  # noqa: E402


def load16(path):
    import librosa

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x


def main():
    model_dir, whisper_path, *qs = sys.argv[1:]
    A = Assistant(model_dir, whisper_path)
    wavs = [load16(q) for q in qs]
    single = []
    for w in wavs:
        t = time.time()
        d = list(A.answer(w))[-1][1]
        single.append((d["text"], d["firstAudioMs"], round(time.time() - t, 1)))
    print("alone: first audio ms", [s[1] for s in single], "| total s", [s[2] for s in single])
    B = Batcher(A, slots=4)

    def run(n, stagger=0.0):
        res, done = [None] * n, [threading.Event() for _ in range(n)]

        def mk(k, t0):
            info = {"first": None, "audio": 0.0, "gaps": 0}

            def emit(kind, v):
                now = time.time()
                if kind == "audio":
                    if info["first"] is None:
                        info["first"] = now
                    elif now - info["first"] > info["audio"] + 0.05:  # playback would have run dry
                        info["gaps"] += 1
                    info["audio"] += len(v) / 24000
                elif kind == "error":
                    print("   ERROR:", v, flush=True)
                elif kind == "done":
                    res[k] = {"text": v.get("text") or "", "first": int(1000 * ((info["first"] or now) - t0)),
                              "gaps": info["gaps"], "audio": round(info["audio"], 1)}
                    done[k].set()
            return emit

        for k in range(n):
            B.submit(AnswerSeq(A, wavs[k % len(wavs)], mk(k, time.time()), lambda: False))
            time.sleep(stagger)
        for d in done:
            d.wait()
        return res

    for n, stagger in ((1, 0), (2, 0), (4, 0), (4, 1.5)):
        res = run(n, stagger)
        same = sum(r["text"] == single[k % len(wavs)][0] for k, r in enumerate(res))
        print(f"{n} questions{' 1.5 s apart' if stagger else ' at once'}: {same}/{n} same text as alone | first audio ms "
              f"{[r['first'] for r in res]} | playback gaps {sum(r['gaps'] for r in res)} | step {B.step_ms:.0f} ms")
        for k, r in enumerate(res):
            if r["text"] != single[k % len(wavs)][0]:
                print(f"   differs: {r['text'][:70]!r}\n   alone:   {single[k % len(wavs)][0][:70]!r}")


if __name__ == "__main__":
    main()
