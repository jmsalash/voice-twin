"""Step 2: general conversation examples from Mini-Omni2's own training set (gpt-omni/VoiceAssistant-400K),
mixed into fine-tuning so the model keeps its general abilities.

The question audio comes from the dataset. The answer is re-spoken with the same Kokoro voice as the CV
answers (--answer-voice), so every training target uses one voice; with --keep-omni-voice the dataset's own
SNAC answer codes (the original Mini-Omni2 voice) are used instead.

  python 02_general.py --n 3000 [--answer-voice af_heart | --keep-omni-voice]
  -> work/general/{questions,answers}/*, work/general/manifest.jsonl
"""

import argparse
import os
import random

import numpy as np
import soundfile as sf

import common as C

DATASET = "datasets/gpt-omni/VoiceAssistant-400K/data"
# --offtopic: the assistant only talks about the person, so these questions get a polite redirect instead.
OFFTOPIC = [
    "Sorry, I can only answer questions about {first}'s professional experience, skills and education.",
    "That's outside what I can help with. I'm here to tell you about {first}'s career, skills and education.",
    "I'm only able to talk about {first}'s professional background. Is there anything you'd like to know about his work?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--answer-voice", default="af_heart")
    ap.add_argument("--keep-omni-voice", action="store_true")
    ap.add_argument("--offtopic", action="store_true", help="answer every general question with a redirect (OFFTOPIC)")
    ap.add_argument("--qa", default=os.path.join(C.HERE, "data", "cv_qa.json"), help="for the person's name (--offtopic)")
    ap.add_argument("--no-answer-audio", action="store_true", help="answers are spoken later by omnivoice_data.py")
    ap.add_argument("--max-answer-words", type=int, default=60, help="skip long answers (the CV answers are short)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rnd = random.Random(args.seed)
    redirects = []
    if args.offtopic:
        import json

        first = json.load(open(args.qa, encoding="utf-8"))["person"].split()[0]
        redirects = [t.format(first=first) for t in OFFTOPIC]
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{DATASET}/*.parquet"))
    rnd.shuffle(files)
    out = os.path.join(C.WORK, "general")
    for d in ("questions", "answers"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    tts = None
    if not args.keep_omni_voice and not args.no_answer_audio:
        from importlib import import_module

        cv_audio = import_module("01_cv_audio")
        tts = cv_audio.kokoro()

    rows = []
    per_file = max(1, args.n // 40)  # spread over many files / sub-datasets
    for f in files:
        if len(rows) >= args.n:
            break
        with fs.open(f, "rb") as fh:
            pf = pq.ParquetFile(fh)
            g = rnd.randrange(pf.num_row_groups)
            t = pf.read_row_group(g, columns=["split_name", "index", "round", "question", "question_audio", "answer", "answer_snac"])
        idx = list(range(t.num_rows))
        rnd.shuffle(idx)
        taken = 0
        for i in idx:
            if taken >= per_file or len(rows) >= args.n:
                break
            r = t.slice(i, 1).to_pylist()[0]
            if not r["answer"] or len(r["answer"].split()) > args.max_answer_words:
                continue
            if r["round"] not in (None, "None", "", "0"):  # single-turn examples only
                continue
            key = f"{r['split_name']}_{r['index']}".replace("/", "_")
            q_path = os.path.join(out, "questions", key + ".wav")
            sf.write(q_path, C.load_audio_16k(r["question_audio"]["bytes"]), 16000)
            row = {"kind": "general", "id": key, "split": "train", "question": r["question"], "question_wav": q_path,
                   "answer": r["answer"]}
            if redirects:
                k = rnd.randrange(len(redirects))
                row.update(kind="offtopic", split="eval" if rnd.random() < 0.03 else "train", answer=redirects[k], answer_wav=os.path.join(out, "answers", f"offtopic_{k}.wav"))
                if not args.no_answer_audio and not os.path.exists(row["answer_wav"]):
                    sf.write(row["answer_wav"], tts.create(row["answer"], voice=args.answer_voice, speed=1.0, lang="en-us")[0], 24000)
            elif args.keep_omni_voice:
                frames = C.frames_from_answer_snac(r["answer_snac"])
                f_path = os.path.join(out, "answers", key + ".frames.npy")
                np.save(f_path, frames)
                row["answer_frames"] = f_path
            elif args.no_answer_audio:
                row["answer_wav"] = os.path.join(out, "answers", key + ".wav")  # not written: omnivoice_data.py speaks it
            else:
                a_path = os.path.join(out, "answers", key + ".wav")
                samples, _ = tts.create(r["answer"], voice=args.answer_voice, speed=1.0, lang="en-us")
                sf.write(a_path, samples, 24000)
                row["answer_wav"] = a_path
            rows.append(row)
            taken += 1
        print(f"{f.split('/')[-1]}: {len(rows)} examples", flush=True)
    C.write_jsonl(os.path.join(out, "manifest.jsonl"), rows)
    print(f"{len(rows)} general examples -> {out}/manifest.jsonl")


if __name__ == "__main__":
    main()
