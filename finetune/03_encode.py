"""Step 3: turn the manifests into training tensors.

For each example: Whisper-small features of the question (T, 768, fp16), the answer's SNAC frames (N, 7)
(from the answer wav, or precomputed frames for --keep-omni-voice), and the answer's text tokens.
Writes work/encoded/{train,eval}.pt (lists of dicts). Run on a GPU machine for speed.

  python 03_encode.py [--manifests work/cv/manifest.jsonl work/general/manifest.jsonl] [--device cuda]
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tokenizers import Tokenizer

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=[os.path.join(C.WORK, "cv", "manifest.jsonl"), os.path.join(C.WORK, "general", "manifest.jsonl")])
    ap.add_argument("--noise", type=float, default=0.5, help="share of training questions given light noise and a gain change")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(C.CKPT_DIR, "tokenizer.json"))
    whisper = C.WhisperFeatures(args.device)
    snac = C.SnacCodec(args.device)
    answer_cache = {}
    rnd = np.random.default_rng(args.seed)

    def augment(wav):
        """Light background noise (15 to 35 dB below the speech) and a gain change, like a cheap microphone."""
        wav = wav * rnd.uniform(0.5, 1.2)
        power = float(np.mean(wav ** 2)) + 1e-9
        snr = rnd.uniform(15, 35)
        noise = rnd.standard_normal(len(wav)).astype(np.float32)
        if rnd.random() < 0.5:  # pinkish noise: more low frequencies
            noise = np.cumsum(noise) - np.convolve(np.cumsum(noise), np.ones(64) / 64, mode="same")
        noise *= np.sqrt(power / 10 ** (snr / 10) / (np.mean(noise ** 2) + 1e-9))
        return np.clip(wav + noise, -1, 1).astype(np.float32)
    data = {"train": [], "eval": []}
    for m in args.manifests:
        if not os.path.exists(m):
            print("skipping missing", m)
            continue
        rows = C.read_jsonl(m)
        for i, r in enumerate(rows):
            q = C.load_audio_16k(r["question_wav"])
            if r["split"] == "train" and rnd.random() < args.noise:
                q = augment(q)
            feat = whisper(q)
            key = r.get("answer_wav") or r.get("answer_frames")
            if key not in answer_cache:
                if r.get("answer_frames"):
                    frames = np.load(r["answer_frames"])
                else:
                    wav, sr = sf.read(r["answer_wav"], dtype="float32")
                    if sr != 24000:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
                    frames = snac.encode(wav)
                answer_cache[key] = frames
            frames = answer_cache[key]
            text_ids = tok.encode(r["answer"]).ids
            data[r["split"]].append({
                "kind": r["kind"], "id": r["id"], "question": r["question"], "answer": r["answer"],
                "feat": feat, "frames": torch.from_numpy(frames.astype(np.int16)), "text": torch.tensor(text_ids, dtype=torch.int32),
                "qids": torch.tensor(tok.encode(r["question"]).ids, dtype=torch.int32),  # for typed (T1A2) rows
                **({"ctx": torch.tensor(tok.encode(r["context_text"]).ids, dtype=torch.int32)} if r.get("context_text") else {}),
            })
            if (i + 1) % 100 == 0:
                print(f"{m}: {i + 1}/{len(rows)}", flush=True)
    out = os.path.join(C.WORK, "encoded")
    os.makedirs(out, exist_ok=True)
    for split, rows in data.items():
        torch.save(rows, os.path.join(out, f"{split}.pt"))
        n_frames = sum(len(r["frames"]) for r in rows)
        print(f"{split}: {len(rows)} examples, {n_frames / 11.72 / 60:.1f} min of answer audio")


if __name__ == "__main__":
    main()
