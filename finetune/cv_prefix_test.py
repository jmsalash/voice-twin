"""Does Mini-Omni2 use a CV given as a text prefix, without any training?

Builds the prompt as [text segment: CV] + [audio segment: spoken question] (both in Mini-Omni2's own
input formats: T1 = [input_t, text..., eot] with audio streams padded, A1 = [input_a, whisper..., eoa]),
then generates greedily like the browser does, and prints the answers next to the no-prefix baseline.

  python cv_prefix_test.py --cv ../cv/cv.txt work/test_questions/*.wav [--device cuda]
"""

import argparse
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

import pipeline as P  # noqa: E402
from omni_lm import (  # noqa: E402
    ANSWER_A, ANSWER_T, EOA, EOT, INPUT_A, INPUT_T, N_AUDIO, PAD_A, PAD_T, embed, layershift, load,
)

torch.set_grad_enabled(False)


def prompt(T, prefix_ids):
    """(2, 8, L) ids and the positions carrying audio features, for rows A1A2 and A1T2."""
    rows = []
    K = len(prefix_ids)
    for last_audio in (ANSWER_A, PAD_A):
        streams = []
        for i in range(N_AUDIO):
            pre = [layershift(PAD_A, i)] * (K + 2) if K else []
            streams.append(pre + [layershift(INPUT_A, i)] + [layershift(PAD_A, i)] * T + [layershift(EOA, i), layershift(last_audio, i)])
        pre_t = [INPUT_T] + prefix_ids + [EOT] if K else []
        streams.append(pre_t + [INPUT_T] + [PAD_T] * T + [EOT, ANSWER_T])
        rows.append(streams)
    ids = torch.tensor(rows)
    mask = torch.zeros(2, ids.shape[-1], dtype=torch.bool)
    start = (K + 2 if K else 0) + 1
    mask[:, start : start + T] = True
    return ids, mask, start


def answer(dec, ad, wte, wav, prefix_ids, device, max_steps=400):
    mel, T = P.load_wav_mel(wav)
    feat = P.whisper_features(mel, T).to(device)
    ids, mask, start = prompt(T, prefix_ids)
    ids, mask = ids.to(device), mask.to(device)
    a = torch.zeros(2, ids.shape[-1], 896, device=device)
    a[:, start : start + T] = ad(feat.unsqueeze(0))[0]
    past = [torch.zeros(2, 2, 0, 64, device=device) for _ in range(48)]

    def step(next_ids, first):
        nonlocal past
        x = embed(wte, ids, a, mask) if first else embed(wte, next_ids.to(device))
        tokens, *past = dec(x, *past)
        return tokens.cpu()

    out = P.generate(step, max_steps)
    return P.decode_text(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--cv", default=os.path.join(HERE, "..", "cv", "cv.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--intro", default="Here is a CV. Answer the user's questions about this person using only this CV.")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(P.CKPT, "tokenizer.json"))
    cv = open(args.cv, encoding="utf-8").read()
    cv = "\n".join(l for l in cv.splitlines() if not re.search(r"\+\d[\d ()-]{6,}\d", l))  # leave phone numbers out
    prefix = tok.encode(args.intro + "\n\n" + " ".join(cv.split())).ids
    print(f"CV prefix: {len(prefix)} tokens")

    dec, ad, wte = load(os.path.join(P.CKPT, "lit_model.pth"))
    dec, ad, wte = dec.to(args.device), ad.to(args.device), wte.to(args.device)
    for wav in args.wavs:
        q = open(wav[:-4] + ".txt", encoding="utf-8").read().strip() if os.path.exists(wav[:-4] + ".txt") else os.path.basename(wav)
        print(f"\nQ: {q}")
        print(f"  no CV  : {answer(dec, ad, wte, wav, [], args.device)}")
        print(f"  CV text: {answer(dec, ad, wte, wav, prefix, args.device)}")


if __name__ == "__main__":
    main()
