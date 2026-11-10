"""Step 1: speak the CV Q&A (data/cv_qa.json) with Kokoro TTS.

Questions: every phrasing x several voices (and speeds), so the model hears many speakers.
Answers: one fixed voice (--answer-voice): this becomes the fine-tuned assistant's voice.
One phrasing per item is held out for evaluation (split "eval").

  python 01_cv_audio.py [--qa data/cv_qa.json] [--voices 6] [--answer-voice af_heart]
  -> work/cv/questions/*.wav (24 kHz), work/cv/answers/*.wav, work/cv/manifest.jsonl
"""

import argparse
import json
import os
import random

import soundfile as sf

import common as C

KOKORO = os.environ.get("KOKORO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "kokoro"))
QUESTION_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def kokoro():
    from kokoro_onnx import Kokoro

    return Kokoro(os.path.join(KOKORO, "kokoro-v1.0.onnx"), os.path.join(KOKORO, "voices-v1.0.bin"))


def speak(tts, text, voice, speed=1.0):
    lang = "en-gb" if voice.startswith("b") else "en-us"
    samples, sr = tts.create(text, voice=voice, speed=speed, lang=lang)
    assert sr == 24000
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=os.path.join(C.HERE, "data", "cv_qa.json"))
    ap.add_argument("--voices", type=int, default=6, help="question voices per phrasing")
    ap.add_argument("--answer-voice", default="af_heart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    ap.add_argument("--extra-questions", default="", help="more training-only phrasings, {id: [questions]} (paraphrase_questions.py)")
    ap.add_argument("--extra-voices", type=int, default=2, help="voices per extra phrasing")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    qa = json.load(open(args.qa, encoding="utf-8"))
    items = qa["items"][: args.limit or None]
    out = os.path.join(C.WORK, "cv")
    os.makedirs(os.path.join(out, "questions"), exist_ok=True)
    os.makedirs(os.path.join(out, "answers"), exist_ok=True)
    tts = kokoro()
    extra = json.load(open(args.extra_questions, encoding="utf-8")) if args.extra_questions else {}
    xrnd = random.Random(args.seed + 1)
    rows = []
    for it in items:
        a_path = os.path.join(out, "answers", f"{it['id']}.wav")
        if not os.path.exists(a_path):
            sf.write(a_path, speak(tts, it["answer"], args.answer_voice), 24000)
        held = rnd.randrange(len(it["questions"]))
        for qi, q in enumerate(it["questions"]):
            split = "eval" if qi == held else "train"
            voices = rnd.sample(QUESTION_VOICES, 1 if split == "eval" else args.voices)
            for v in voices:
                speed = 1.0 if split == "eval" else rnd.choice([0.9, 1.0, 1.0, 1.1, 1.2])
                q_path = os.path.join(out, "questions", f"{it['id']}_{qi}_{v}_{speed}.wav")
                if not os.path.exists(q_path):
                    sf.write(q_path, speak(tts, q, v, speed), 24000)
                rows.append({"kind": "cv", "id": it["id"], "split": split, "question": q, "question_wav": q_path,
                             "answer": it["answer"], "answer_wav": a_path, "voice": v})
        for qi, q in enumerate(extra.get(it["id"], [])):
            # a separate random stream, so adding extras doesn't change the voices and held-out picks above
            for v in xrnd.sample(QUESTION_VOICES, args.extra_voices):
                speed = xrnd.choice([0.9, 1.0, 1.0, 1.1, 1.2])
                q_path = os.path.join(out, "questions", f"{it['id']}_x{qi}_{v}_{speed}.wav")
                if not os.path.exists(q_path):
                    sf.write(q_path, speak(tts, q, v, speed), 24000)
                rows.append({"kind": "cv", "id": it["id"], "split": "train", "question": q, "question_wav": q_path,
                             "answer": it["answer"], "answer_wav": a_path, "voice": v, "extra": True})
        print(f"{it['id']}: {sum(r['id'] == it['id'] for r in rows)} question clips", flush=True)
    C.write_jsonl(os.path.join(out, "manifest.jsonl"), rows)
    print(f"{len(rows)} examples ({sum(r['split'] == 'eval' for r in rows)} eval) -> {out}/manifest.jsonl")


if __name__ == "__main__":
    main()
