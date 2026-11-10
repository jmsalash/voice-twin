"""Would a "please rephrase" safety net catch the wrong answers?

At inference Mini-Omni2 runs two rows: A1T2 writes the text, A1A2 speaks and also predicts text of its own, which
is normally thrown away. When the two rows would pick different answers, the answer is often wrong (and the voice,
forced to follow the other row's text, garbles). This generates the held-out questions, records both rows' text,
and reports how many wrong answers a check on the first words would catch, and how many right ones it would block.

  python agreement_test.py --ckpt work/ckpt_v3 [--words 6]
"""

import argparse
import json
import os
import sys

import torch

import common as C
from score_eval import wer, words

sys.path.insert(0, os.path.join(C.HERE, "..", "model"))
import pipeline as P  # noqa: E402
from omni_lm import EOT, N_AUDIO, PAD_T, embed, empty_past, load  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--words", type=int, default=6, help="compare the first N words of both rows")
    ap.add_argument("--eval", default=os.path.join(C.WORK, "encoded", "eval.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(C.CKPT_DIR, "tokenizer.json"))
    rows = torch.load(args.eval, weights_only=False)
    dec, ad, wte = load(os.path.join(args.ckpt, "lit_model.pth"))
    dec, ad, wte = dec.to(args.device), ad.to(args.device), wte.to(args.device)
    res = []
    for r in rows:
        if r["kind"] != "cv":
            continue
        feat = r["feat"].float().to(args.device)
        T = feat.shape[0]
        ids, mask = P.prompt_ids(T)
        ids, mask = ids.to(args.device), mask.to(args.device)
        a = torch.zeros(2, ids.shape[-1], 896, device=args.device)
        a[:, 1 : T + 1] = ad(feat.unsqueeze(0))[0]
        past = [p.to(args.device) for p in empty_past(2)]
        own = []  # the speaking row's own text predictions

        def step(next_ids, first):
            nonlocal past
            x = embed(wte, ids, a, mask) if first else embed(wte, next_ids.to(args.device))
            tokens, *past = dec(x, *past)
            t = int(tokens[0, N_AUDIO])
            if not own or own[-1] != EOT:
                own.append(t)
            return tokens.cpu()

        out = P.generate(step, 600)
        text = P.decode_text(out)
        own_text = tok.decode([t for t in own if t < 151936 and t not in (EOT, PAD_T)]).strip()
        n = args.words
        agree = words(text)[:n] == words(own_text)[:n]
        res.append({"id": r["id"], "question": r["question"], "right": wer(r["answer"], text) <= 0.1,
                    "agree": agree, "text": text, "own": own_text})
    right = [x for x in res if x["right"]]
    wrong = [x for x in res if not x["right"]]
    caught = [x for x in wrong if not x["agree"]]
    blocked = [x for x in right if not x["agree"]]
    print(f"{len(res)} CV questions: {len(right)} right, {len(wrong)} wrong")
    print(f"rows disagree (first {args.words} words) on {len(caught)}/{len(wrong)} wrong answers -> 'please rephrase'")
    print(f"rows disagree on {len(blocked)}/{len(right)} right answers -> blocked unnecessarily")
    print(f"=> spoken: {len(right) - len(blocked)} right, {len(wrong) - len(caught)} wrong, "
          f"{len(caught) + len(blocked)} asked to rephrase")
    for x in wrong:
        if x["agree"]:
            print(f"  missed [{x['id']}] {x['question']!r} -> {x['text'][:80]!r}")
    C.write_jsonl(os.path.join(args.ckpt, "eval", "agreement.jsonl"), res)


if __name__ == "__main__":
    main()
