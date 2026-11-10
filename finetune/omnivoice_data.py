"""Optional step between 2 and 3: re-speak the training audio with k2-fsa OmniVoice instead of Kokoro.

* Answers (CV and general): every answer is spoken in ONE voice, cloned from a reference clip (--ref, with its
  words in a .txt next to it). That voice becomes the fine-tuned assistant's voice. Each clip is checked with
  Whisper and generated again (new seed) when the words don't match.
* Extra CV questions: --question-voices more clips per training phrasing, each in a new voice designed from a random
  description (gender, age, pitch, English accent), so the model hears more kinds of speakers.

Writes the new clips next to the old ones (answers_ov/, questions_ov/) and new manifests (manifest.ov.jsonl):
encode with 03_encode.py --manifests work/cv/manifest.ov.jsonl work/general/manifest.ov.jsonl. Needs a Python with the omnivoice package (pip install omnivoice).
OmniVoice's weights are CC-BY-NC: non-commercial use only.

  python omnivoice_data.py --ref voice_ref.wav [--question-voices 2] [--steps 32]
"""

import argparse
import json
import os
import random
import re
import time
import unicodedata
import zlib

import numpy as np
import soundfile as sf
import torch

WORK = os.environ.get("FT_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
GENDERS = ["male", "female"]
AGES = ["teenager", "young adult", "middle-aged", "elderly"]
PITCHES = ["low pitch", "moderate pitch", "high pitch"]
ACCENTS = ["american accent", "british accent", "australian accent", "canadian accent", "indian accent", None]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


# words Whisper writes differently from the answer text but that sound the same (accents are already ignored)
SAME = {"colour": "color"}


def words(s):
    s = unicodedata.normalize("NFKD", s.lower().replace("&", " and ").replace("%", " percent"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in SAME.items():
        s = s.replace(unicodedata.normalize("NFKD", a).encode("ascii", "ignore").decode(), b)
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
    ap.add_argument("--ref", required=True, help="reference clip of the answer voice; its words in REF.txt")
    ap.add_argument("--question-voices", type=int, default=2, help="extra OmniVoice clips per CV training phrasing")
    ap.add_argument("--steps", type=int, default=32, help="OmniVoice decoding steps (16 is ~2x faster)")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking rate of the answer voice (e.g. 0.9)")
    ap.add_argument("--max-wer", type=float, default=0.15, help="regenerate clips whose transcript differs more")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pronounce", action="append", default=[], help="WRITTEN=SPOKEN for the answer voice, e.g. Siobhan=Shivawn")
    ap.add_argument("--jobs", default="", help="only speak [{text, path}] in the answer voice (remap_topics.py)")
    args = ap.parse_args()

    from omnivoice import OmniVoice

    rnd = random.Random(args.seed)
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float32)
    ref_text = open(os.path.splitext(args.ref)[0] + ".txt", encoding="utf-8").read().strip()
    # the voice prompt first: encoding a long reference clip next to Whisper does not fit on an 8 GB card
    prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=ref_text)
    torch.cuda.empty_cache()
    m.load_asr_model()  # Whisper large-v3-turbo, fp16: checks every clip
    stats = {"clips": 0, "retried": 0, "failed": 0, "audio_s": 0.0}
    speed = {"speed": args.speed} if args.speed != 1.0 else {}
    pron = dict(p.split("=", 1) for p in args.pronounce)

    def say(text):  # how names should sound: changes the TTS text only
        for a, b in pron.items():
            text = text.replace(a, b)
        return text
    t0 = time.time()

    def transcribe(a, sr=24000):
        """Whisper reads at most 30 s at once: longer clips are cut in two at the quietest point near the middle."""
        if len(a) <= 29 * sr:
            return m.transcribe((torch.from_numpy(a).unsqueeze(0), sr))
        lo, hi, win = len(a) // 2 - 3 * sr, len(a) // 2 + 3 * sr, sr // 20
        energy = [float(np.abs(a[k:k + win]).mean()) for k in range(lo, hi, win)]
        cut = lo + int(np.argmin(energy)) * win + win // 2
        return transcribe(a[:cut], sr) + " " + transcribe(a[cut:], sr)

    def make(text, path, keep=False, **voice):
        """Speak text to path (skipped if it exists); keep the attempt whose transcript matches best. Without keep, a
        clip that never matches is dropped. Answers are always kept: names and addresses are often only spelled
        differently by Whisper, and dropping an answer would drop its whole topic."""
        if os.path.exists(path):
            return True
        best = None
        limit = max(args.max_wer, 1.01 / max(1, len(words(text))))  # short texts: one wrong word is allowed
        for k in range(args.tries):
            torch.manual_seed(args.seed + k * 1000 + stats["clips"])
            try:
                a = np.asarray(m.generate(text=text, language="en", num_step=args.steps, **voice)[0]).squeeze()
            except ValueError:  # OmniVoice occasionally produces an empty clip
                stats["retried"] += 1
                continue
            if a.size < 2400:
                stats["retried"] += 1
                continue
            hyp = transcribe(a)
            e = wer(text, hyp)
            if best is None or e < best[0]:
                best = (e, a, hyp)
            if e <= limit:
                break
            stats["retried"] += 1
        stats["clips"] += 1
        if best is None:
            stats["failed"] += 1
            print(f"  no audio: {text[:70]!r}", flush=True)
            return False
        if best[0] > limit and keep:
            print(f"  kept anyway (WER {best[0]:.2f}): {text[:70]!r} heard {best[2][:70]!r}", flush=True)
        elif best[0] > limit:
            stats["failed"] += 1
            print(f"  dropped (WER {best[0]:.2f}): {text[:70]!r} heard {best[2][:70]!r}", flush=True)
            return False
        sf.write(path, best[1], 24000)
        stats["audio_s"] += len(best[1]) / 24000
        if stats["clips"] % 50 == 0:
            el = time.time() - t0
            print(f"  {stats['clips']} clips, {stats['audio_s'] / 60:.0f} min audio in {el / 60:.0f} min "
                  f"({stats['audio_s'] / el:.2f}x), {stats['retried']} retries, {stats['failed']} dropped", flush=True)
        return True

    if args.jobs:
        jobs = json.load(open(args.jobs, encoding="utf-8"))
        for j in jobs:
            make(j["text"], j["path"], keep=True, voice_clone_prompt=prompt, **speed)
        print(f"done: {len(jobs)} answers, {stats['retried']} retries")
        return

    for kind in ("cv", "general"):
        man = os.path.join(WORK, kind, "manifest.jsonl")
        if not os.path.exists(man):
            print("skipping missing", man)
            continue
        rows = read_jsonl(man)
        os.makedirs(os.path.join(WORK, kind, "answers_ov"), exist_ok=True)
        os.makedirs(os.path.join(WORK, kind, "questions_ov"), exist_ok=True)

        # answers, one per distinct answer clip
        ok = {}
        for r in rows:
            src = r.get("answer_wav") or os.path.join(WORK, kind, "answers", r["id"] + ".wav")
            dst = os.path.join(WORK, kind, "answers_ov", os.path.basename(src))
            if dst not in ok:
                ok[dst] = make(say(r["answer"]), dst, keep=True, voice_clone_prompt=prompt, **speed)
            r["answer_wav"] = dst
        rows = [r for r in rows if ok[r["answer_wav"]]]
        print(f"{kind}: {sum(ok.values())}/{len(ok)} answers", flush=True)

        # extra question voices (CV training phrasings only)
        if kind == "cv" and args.question_voices:
            extra, seen = [], set()
            for r in rows:
                key = (r["id"], r["question"])
                if r["split"] != "train" or r.get("extra") or key in seen:  # extras (paraphrases) keep their Kokoro voices
                    continue
                seen.add(key)
                for v in range(args.question_voices):
                    desc = ", ".join(x for x in (rnd.choice(GENDERS), rnd.choice(AGES), rnd.choice(PITCHES), rnd.choice(ACCENTS)) if x)
                    q_path = os.path.join(WORK, kind, "questions_ov", f"{r['id']}_{zlib.crc32(r['question'].encode())}_{v}.wav")
                    if make(r["question"], q_path, instruct=desc):
                        extra.append({**r, "question_wav": q_path, "voice": "omnivoice: " + desc})
            rows += extra
            print(f"cv: {len(extra)} extra question clips", flush=True)
        write_jsonl(os.path.join(WORK, kind, "manifest.ov.jsonl"), rows)

    el = time.time() - t0
    print(f"done: {stats['clips']} clips, {stats['audio_s'] / 3600:.1f} h of audio in {el / 3600:.1f} h, "
          f"{stats['retried']} retries, {stats['failed']} dropped")


if __name__ == "__main__":
    main()
