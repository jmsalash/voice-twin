"""Fine-tunes the retriever's embedding model on the Q&A wordings, so wordings of one topic sit together.

Contrastive training (in-batch negatives): each batch holds pairs of different wordings of the same topic, one pair
per topic, and the model learns to match each wording with its partner rather than with the other topics' wordings.
Only training wordings are used; the held-out ones stay unseen, so the evaluation is honest.

  python train_retriever.py --manifest work/cv/manifest.v4.jsonl --out work/retriever [--epochs 3]
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "assistant"))
from retrieval import Retriever, _mean_pool, load_wordings  # noqa: E402


def evaluate(R, wordings, evals):
    R.build(wordings)
    top1 = top2 = 0
    for q, t in evals:
        res = [x for x, _ in R.search(q, k=2)]
        top1 += res[0] == t
        top2 += t in res
    return top1, top2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--extra-eval", default="", help="json [[question, topic]] also evaluated (e.g. transcripts)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    torch.manual_seed(args.seed)

    wordings = load_wordings(args.manifest)
    evals = [(r["question"], r["id"]) for r in map(json.loads, open(args.manifest, encoding="utf-8")) if r["split"] == "eval"]
    extra = json.load(open(args.extra_eval)) if args.extra_eval else []
    by_topic = {}
    for q, t in wordings:
        by_topic.setdefault(t, []).append(q)
    R = Retriever(args.base, device="cuda")
    t1, t2 = evaluate(R, wordings, evals)
    print(f"before: held-out top-1 {t1}/{len(evals)}, top-2 {t2}/{len(evals)}", flush=True)
    if extra:
        e1, e2 = evaluate(R, wordings, extra)
        print(f"        transcripts top-1 {e1}/{len(extra)}, top-2 {e2}/{len(extra)}", flush=True)

    opt = torch.optim.AdamW(R.emb.parameters(), lr=args.lr)
    topics = [t for t, qs in by_topic.items() if len(qs) >= 2]
    for ep in range(args.epochs):
        R.emb.train()
        # every wording is an anchor once per epoch; each batch has one pair per topic (no false negatives)
        anchors = [(q, t) for t in topics for q in by_topic[t]]
        rnd.shuffle(anchors)
        batches, cur, used = [], [], set()
        for q, t in anchors:
            if t in used:
                continue
            p = rnd.choice([x for x in by_topic[t] if x != q])
            cur.append((q, p))
            used.add(t)
            if len(cur) == len(topics) // 2:
                batches.append(cur)
                cur, used = [], set()
        total = 0.0
        for b in batches:
            enc = lambda texts: R.tok(["query: " + x for x in texts], padding=True, truncation=True, max_length=64,  # noqa: E731
                                      return_tensors="pt").to("cuda")
            ea, ep_ = enc([a for a, _ in b]), enc([p for _, p in b])
            za = F.normalize(_mean_pool(R.emb(**ea).last_hidden_state, ea["attention_mask"]), dim=-1)
            zp = F.normalize(_mean_pool(R.emb(**ep_).last_hidden_state, ep_["attention_mask"]), dim=-1)
            logits = za @ zp.T / args.temperature
            labels = torch.arange(len(b), device="cuda")
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        R.emb.eval()
        t1, t2 = evaluate(R, wordings, evals)
        line = f"epoch {ep + 1}: loss {total / max(1, len(batches)):.3f}, held-out top-1 {t1}/{len(evals)}, top-2 {t2}/{len(evals)}"
        if extra:
            e1, e2 = evaluate(R, wordings, extra)
            line += f" | transcripts top-1 {e1}/{len(extra)}, top-2 {e2}/{len(extra)}"
        print(line, flush=True)
    os.makedirs(args.out, exist_ok=True)
    R.emb.save_pretrained(args.out)
    R.tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
