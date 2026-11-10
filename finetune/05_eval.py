"""Step 5: ask a checkpoint the held-out CV questions (and a few general ones) and show the spoken answers' text.

Uses the same greedy two-row generation as the browser (mini-omni2-web: export/pipeline.py) with the PyTorch port.
Optionally writes the answers' audio (--wav) so you can listen to the fine-tuned voice.

  python 05_eval.py --ckpt work/ckpt [--n 20] [--wav]
"""

import argparse
import os
import sys

import numpy as np
import torch

import common as C

sys.path.insert(0, os.path.join(C.HERE, "..", "model"))
import pipeline as P  # noqa: E402
from omni_lm import embed, empty_past, load  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(C.WORK, "ckpt"))
    ap.add_argument("--n", type=int, default=0, help="max eval questions (0 = all)")
    ap.add_argument("--wav", action="store_true")
    ap.add_argument("--rag", action="store_true", help="put each example's retrieved answers (ctx) in front, as in training")
    ap.add_argument("--out", default="", help="results folder (default CKPT/eval, or CKPT/eval_rag with --rag)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = torch.load(os.path.join(C.WORK, "encoded", "eval.pt"), weights_only=False)[: args.n or None]
    dec, ad, wte = load(os.path.join(args.ckpt, "lit_model.pth"))
    dec, ad, wte = dec.to(args.device), ad.to(args.device), wte.to(args.device)
    snac = C.SnacCodec(args.device) if args.wav else None
    out_dir = args.out or os.path.join(args.ckpt, "eval_rag" if args.rag else "eval")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for n, r in enumerate(rows):
        feat = r["feat"].float().to(args.device)
        T = feat.shape[0]
        ids, mask = P.prompt_ids(T)
        off = 0
        if args.rag:  # the retrieved answers as a text segment before the question (common.build_sequence's layout)
            ctx = r["ctx"].tolist()
            pre = torch.tensor([[C.layershift(C.PAD_A, i)] * (len(ctx) + 2) for i in range(7)] + [[C.INPUT_T] + ctx + [C.EOT]])
            ids = torch.cat([pre.unsqueeze(0).expand(2, -1, -1), ids], -1)
            off = pre.shape[1]
            mask = torch.cat([torch.zeros(2, off, dtype=torch.bool), mask], -1)
        ids, mask = ids.to(args.device), mask.to(args.device)
        a = torch.zeros(2, ids.shape[-1], 896, device=args.device)
        a[:, off + 1 : off + T + 1] = ad(feat.unsqueeze(0))[0]
        past = empty_past(2)
        past = [p.to(args.device) for p in past]

        def step(next_ids, first):
            nonlocal past
            x = embed(wte, ids, a, mask) if first else embed(wte, next_ids.to(args.device))
            tokens, *past = dec(x, *past)
            return tokens.cpu()

        out = P.generate(step, 600)
        text = P.decode_text(out)
        print(f"\nQ: {r['question']}\n  expected: {r['answer']}\n  model:    {text}", flush=True)
        res = {"id": r["id"], "kind": r["kind"], "question": r["question"], "expected": r["answer"], "model": text}
        if snac is not None:
            N = len(out[6]) - 7
            frames = np.stack([np.array(out[i][i + 1 : i + 1 + N]) for i in range(7)], 1).clip(0, 4095)
            import soundfile as sf

            res["wav"] = os.path.join(out_dir, f"{n:03d}_{r['id']}.wav")
            sf.write(res["wav"], snac.decode(frames), 24000)
        results.append(res)
    C.write_jsonl(os.path.join(out_dir, "results.jsonl"), results)
    print("written", os.path.join(out_dir, "results.jsonl"))


if __name__ == "__main__":
    main()
