"""End-to-end RAG evaluation, as the server will run it: transcript -> retrieval -> spoken answer.

For each held-out question: its Whisper transcript (work/eval_transcripts.json) is looked up with the retriever,
the top-k answers go in front of the spoken question, and the answer is generated. Records, for the precision
curve (precision_curve.py): the answer text, the speaking row's own text (safety net), the retrieved topics and
their scores (retriever confidence).

  python rag_eval.py --ckpt work/ckpt_v5 [--retriever work/retriever] [--k 2]
  -> CKPT/eval_rag_e2e/results.jsonl
"""

import argparse
import json
import os
import sys

import torch

import common as C
from score_eval import wer

sys.path.insert(0, os.path.join(C.HERE, "..", "model"))
sys.path.insert(0, os.path.join(C.HERE, "..", "assistant"))
import pipeline as P  # noqa: E402
from omni_lm import EOT, N_AUDIO, PAD_T, embed, empty_past, load  # noqa: E402
from retrieval import Retriever, load_wordings  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--retriever", default=os.path.join(C.WORK, "retriever"))
    ap.add_argument("--manifest", default=os.path.join(C.WORK, "cv", "manifest.v4.jsonl"))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(C.CKPT_DIR, "tokenizer.json"))
    answers = {r["id"]: r["answer"] for r in C.read_jsonl(args.manifest)}
    R = Retriever(args.retriever, device="cuda").build(load_wordings(args.manifest))
    heard = {o: q for q, _, o in json.load(open(os.path.join(C.WORK, "eval_transcripts.json")))}
    rows = torch.load(os.path.join(C.WORK, "encoded", "eval.pt"), weights_only=False)
    dec, ad, wte = load(os.path.join(args.ckpt, "lit_model.pth"))
    dec, ad, wte = dec.cuda(), ad.cuda(), wte.cuda()
    out_dir = args.out or os.path.join(args.ckpt, "eval_rag_e2e")
    os.makedirs(out_dir, exist_ok=True)
    res = []
    for r in rows:
        q = heard.get(r["question"], r["question"])  # off-topic questions: the dataset's own text
        found = R.search(q, k=args.k)
        ctx = tok.encode("\n".join(answers[t] for t, _ in found)).ids
        feat = r["feat"].float().cuda()
        T = feat.shape[0]
        ids, mask = P.prompt_ids(T)
        pre = torch.tensor([[C.layershift(C.PAD_A, i)] * (len(ctx) + 2) for i in range(7)] + [[C.INPUT_T] + ctx + [C.EOT]])
        ids = torch.cat([pre.unsqueeze(0).expand(2, -1, -1), ids], -1).cuda()
        off = pre.shape[1]
        mask = torch.cat([torch.zeros(2, off, dtype=torch.bool), mask], -1).cuda()
        a = torch.zeros(2, ids.shape[-1], 896, device="cuda")
        a[:, off + 1: off + T + 1] = ad(feat.unsqueeze(0))[0]
        past = [p.cuda() for p in empty_past(2)]
        own = []

        def step(next_ids, first):
            nonlocal past
            x = embed(wte, ids, a, mask) if first else embed(wte, next_ids.cuda())
            tokens, *past = dec(x, *past)
            t = int(tokens[0, N_AUDIO])
            if not own or own[-1] != EOT:
                own.append(t)
            return tokens.cpu()

        gen = P.generate(step, 700)
        text = P.decode_text(gen)
        own_text = tok.decode([t for t in own if t < 151936 and t not in (EOT, PAD_T)]).strip()
        # how much of the answer the voice covers: audio frames vs. a rough 12 frames per second of speech
        res.append({"id": r["id"], "kind": r["kind"], "question": r["question"], "heard": q, "expected": r["answer"],
                    "model": text, "own": own_text, "topics": [t for t, _ in found],
                    "scores": [round(s, 4) for _, s in found], "frames": len(gen[6]) - 7,
                    "right_exact": wer(r["answer"], text) <= 0.1})
        print(f"{r['question'][:50]!r} -> {text[:80]!r}", flush=True)
    C.write_jsonl(os.path.join(out_dir, "results.jsonl"), res)
    print("written", os.path.join(out_dir, "results.jsonl"))


if __name__ == "__main__":
    main()
