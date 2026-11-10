"""Scores 05_eval.py's results: did the model say the expected answer, and does its speech say its own text?

  python score_eval.py [work/ckpt/eval/results.jsonl] [--whisper small.pt]

* text: word error rate of the model's text against the expected answer. Right = WER <= 0.1.
* speech: Whisper's transcript of the generated audio against the model's own text (needs 05_eval.py --wav).
"""

import argparse
import json
import os
import re
import unicodedata

import common as C


def words(s):
    s = unicodedata.normalize("NFKD", s.lower().replace("&", " and "))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("suisse", "swiss").replace("mini-omni2", "mini omni 2").replace("mini omni2", "mini omni 2")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join(C.WORK, "ckpt", "eval", "results.jsonl"))
    ap.add_argument("--whisper", default=os.path.join(C.CKPT_DIR, "small.pt"))
    ap.add_argument("--skip", default="", help="comma-separated ids not trained in this checkpoint")
    ap.add_argument("--acceptable", default="", help="json {question: [other topic ids]} also counted as right")
    ap.add_argument("--answers", default="", help="manifest giving each topic's answer (needed with --acceptable)")
    args = ap.parse_args()
    rows = C.read_jsonl(args.results)
    skip = set(filter(None, args.skip.split(",")))
    asr = None
    if any(r.get("wav") for r in rows):
        import whisper

        asr = whisper.load_model(args.whisper)
    by_kind = {}
    alt = json.load(open(args.acceptable, encoding="utf-8")) if args.acceptable else {}
    answer_of = {x["id"]: x["answer"] for x in C.read_jsonl(args.answers)} if args.answers else {}
    for r in rows:
        r["text_wer"] = round(wer(r["expected"], r["model"]), 3)
        for t in alt.get(r["question"], []):  # another answer that also answers the question
            r["text_wer"] = min(r["text_wer"], round(wer(answer_of[t], r["model"]), 3))
        if asr and r.get("wav"):
            r["heard"] = asr.transcribe(r["wav"], language="en")["text"].strip()
            r["speech_wer"] = round(wer(r["model"], r["heard"]), 3)
        kind = "untrained" if r["id"] in skip else r["kind"]
        by_kind.setdefault(kind, []).append(r)
    for kind, rs in by_kind.items():
        right = [r for r in rs if r["text_wer"] <= 0.1]
        line = f"{kind}: {len(right)}/{len(rs)} right answers"
        if asr:
            sw = [r["speech_wer"] for r in rs if "speech_wer" in r]
            line += f", speech matches its text: mean WER {sum(sw) / max(1, len(sw)):.3f}, {sum(w <= 0.1 for w in sw)}/{len(sw)} clips within 0.1"
        print(line)
        for r in rs:
            if r["text_wer"] > 0.1 or r.get("speech_wer", 0) > 0.1:
                print(f"  [{r['id']}] Q: {r['question']}\n     expected: {r['expected']}\n     model:    {r['model']}"
                      + (f"\n     heard:    {r['heard']}  (speech WER {r['speech_wer']})" if "heard" in r else ""))
    out = os.path.splitext(args.results)[0] + ".scored.jsonl"
    C.write_jsonl(out, rows)
    print("written", out)


if __name__ == "__main__":
    main()
