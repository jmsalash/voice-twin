"""Checks the training format and model against the original: for answers that the original model generated
greedily (mini-omni2-web: export/out/reference/*.npy; set WEB_EXPORT_DIR), rebuild them as training sequences (common.build_sequence) and
teacher-force them through OmniTrain with the original weights. Every target must equal the argmax."""

import os
import sys

import numpy as np
import torch

import common as C
from omni_train import OmniTrain

sys.path.insert(0, os.path.join(C.HERE, "..", "model"))
# the browser port's reference outputs (export/run.py in mini-omni2-web)
WEB_EXPORT = os.environ.get("WEB_EXPORT_DIR", os.path.join(C.HERE, "..", "..", "mini-omni2-web", "export"))
import pipeline as P  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = OmniTrain().load_lit(os.path.join(C.CKPT_DIR, "lit_model.pth")).to(dev).eval()
wf = C.WhisperFeatures(dev)
ok = True
for name in sys.argv[1:] or ["output1", "output4"]:
    out = np.load(os.path.join(WEB_EXPORT, "out", "reference", name + ".npy"))  # (8, S)
    # generation stops when stream 6 emits EOA, without storing that step: N frames -> N + 7 stored steps
    N = out.shape[1] - 7
    frames = np.stack([out[i, i + 1 : i + 1 + N] for i in range(7)], 1)  # undo the delay
    text = out[7].tolist()
    text = text[: text.index(C.EOT)]
    feat = wf(C.load_audio_16k(os.path.join(WEB_EXPORT, "samples", name + ".wav")))
    T = feat.shape[0]
    for row in ("A1A2", "A1T2"):
        ids, mask, tgt = C.build_sequence(T, text, frames, row)
        L = T + 3
        if row == "A1A2":
            t = tgt[:, L - 1 : L - 1 + out.shape[1]]
            keep = t != C.IGNORE
            assert np.array_equal(t[keep], out[keep]), "rebuilt targets differ from the generated answer"
        f = torch.zeros(1, ids.shape[1], 768)
        f[0, 1 : T + 1] = feat.float()
        with torch.no_grad():
            h = model(torch.from_numpy(ids)[None].to(dev), f.to(dev), torch.from_numpy(mask)[None].to(dev))
            pred = model.greedy_targets(h)[0].cpu().numpy()
            loss = model.losses(h, torch.from_numpy(tgt)[None].to(dev))
        sel = tgt != C.IGNORE
        if row == "A1A2":
            # Row A1A2 provides the audio at inference (its text output is never used: the text comes from
            # row A1T2), so its audio streams must match exactly, including the final EOA of stream 6.
            agree = (pred[:7][sel[:7]] == tgt[:7][sel[:7]]).mean()
            text_agree = (pred[7][sel[7]] == tgt[7][sel[7]]).mean()
            print(f"  (A1A2 row's own text predictions, unused at inference: {text_agree * 100:.1f}% agree)")
        else:
            agree = (pred[7][sel[7]] == tgt[7][sel[7]]).mean()
        ok &= agree == 1.0
        print(f"{name} {row}: {sel.sum()} targets, argmax agreement {agree * 100:.2f}%, "
              f"loss text {loss['text'].item():.4f}" + (f", audio0 {loss['audio0'].item():.4f}" if 'audio0' in loss else ""))

# Typed questions (T1A2, one row gives text and audio): the original model's own greedy answers, rebuilt with
# build_sequence(..., "T1A2", q_ids) and teacher-forced, must give back every token.
from omni_lm import embed, empty_past, load  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(os.path.join(C.CKPT_DIR, "tokenizer.json"))
dec, _, wte = load(os.path.join(C.CKPT_DIR, "lit_model.pth"))
dec, wte = dec.to(dev), wte.to(dev)
for question in ["What is the capital of France?", "Tell me a fun fact about the ocean."]:
    q = tok.encode(question).ids
    streams = [[C.layershift(C.PAD_A, i)] * (len(q) + 2) + [C.layershift(C.ANSWER_A, i)] for i in range(7)]
    streams.append([C.INPUT_T] + q + [C.EOT, C.ANSWER_T])
    with torch.no_grad():
        tokens, *past = dec(embed(wte, torch.tensor([streams], device=dev)), *[p.to(dev) for p in empty_past(1)])
        gen = [[] for _ in range(8)]
        text_end = False
        for k in range(600):
            audio, text = tokens[0, :7].tolist(), int(tokens[0, 7])
            if text_end:
                text = C.PAD_T
            if k > 0 and audio[-1] == C.EOA:
                break
            if text == C.EOT:
                text_end = True
            for i in range(7):
                gen[i].append(audio[i])
            gen[7].append(text)
            nxt = torch.tensor([[[C.layershift(a, i)] for i, a in enumerate(audio)] + [[text]]], device=dev)
            tokens, *past = dec(embed(wte, nxt), *past)
    out = np.array(gen)
    N = out.shape[1] - 7
    frames = np.stack([out[i, i + 1 : i + 1 + N] for i in range(7)], 1)
    text = out[7].tolist()
    text = text[: text.index(C.EOT)]
    ids, mask, tgt = C.build_sequence(0, text, frames, "T1A2", q_ids=q)
    L = len(q) + 3
    t = tgt[:, L - 1 : L - 1 + out.shape[1]]
    keep = t != C.IGNORE
    assert np.array_equal(t[keep], out[keep]), "rebuilt targets differ from the generated answer"
    with torch.no_grad():
        h = model(torch.from_numpy(ids)[None].to(dev), torch.zeros(1, ids.shape[1], 768, device=dev), torch.from_numpy(mask)[None].to(dev))
        pred = model.greedy_targets(h)[0].cpu().numpy()
    sel = tgt != C.IGNORE
    agree = (pred[sel] == tgt[sel]).mean()
    ok &= agree == 1.0
    print(f"typed {question!r} T1A2: {tok.decode(text)!r}, {N} frames, {sel.sum()} targets, argmax agreement {agree * 100:.2f}%")
print("FORMAT OK" if ok else "FORMAT MISMATCH")
