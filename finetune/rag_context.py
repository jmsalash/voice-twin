"""Adds retrieved answers (RAG context) to every example, for training the model to read and speak the right one.

For each question, the top-k topics from assistant/retrieval.py (the index holds the training wordings) are put in
front of it as text. A training question's own wording is left out of the index when it is looked up, so the
training sees retrieval as it will be for new wordings, mistakes included: sometimes the right answer is second,
sometimes missing. Off-topic questions get CV answers too, and are still trained to redirect, so the model learns
not to just read out whatever was retrieved.

  python rag_context.py --manifests work/cv/manifest.v4.jsonl work/general/manifest.v4.jsonl [--k 2] [--rerank MODEL]
  -> manifest.rag.jsonl next to each (rows gain "context": [topic ids] and "context_text")
"""

import argparse
import os
import sys

import common as C

sys.path.insert(0, os.path.join(C.HERE, "..", "assistant"))
from retrieval import Retriever, load_wordings  # noqa: E402

SEP = "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True, help="the first one (CV) builds the index")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--rerank", default="")
    ap.add_argument("--retriever", default="intfloat/multilingual-e5-base", help="embedding model (a fine-tuned one: train_retriever.py)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--patch-encoded", default="", help="add the context to an encoded dir (03_encode output of the same "
                    "manifests, same order) instead of re-encoding")
    args = ap.parse_args()

    wordings = load_wordings(args.manifests[0])
    answers = {r["id"]: r["answer"] for r in C.read_jsonl(args.manifests[0])}
    R = Retriever(args.retriever, rerank_model=args.rerank or None, device=args.device).build(wordings)
    cache, all_rows = {}, []
    for man in args.manifests:
        rows = C.read_jsonl(man)
        hit = n = 0
        for r in rows:
            key = (r["question"], r["split"])
            if key not in cache:
                own = r["question"] if r["split"] == "train" and r["kind"] == "cv" else None
                cache[key] = [t for t, _ in R.search(r["question"], k=args.k, exclude=own)]
            r["context"] = cache[key]
            r["context_text"] = SEP.join(answers[t] for t in r["context"])
            if r["kind"] == "cv":
                n += 1
                hit += r["id"] in r["context"]
        out = man.replace(".jsonl", ".rag.jsonl") if ".v4." not in man else man.replace(".v4.jsonl", ".rag.jsonl")
        C.write_jsonl(out, rows)
        all_rows += rows
        print(f"{out}: {len(rows)} examples" + (f", right answer in the context for {hit}/{n} CV examples" if n else ""), flush=True)

    if args.patch_encoded:
        import torch
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(C.CKPT_DIR, "tokenizer.json"))
        enc = {sp: torch.load(os.path.join(args.patch_encoded, f"{sp}.pt"), weights_only=False) for sp in ("train", "eval")}
        pos = {"train": 0, "eval": 0}
        for r in all_rows:  # 03_encode appends rows to their split in manifest order
            e = enc[r["split"]][pos[r["split"]]]
            pos[r["split"]] += 1
            assert e["question"] == r["question"] and e["id"] == r["id"], f"order mismatch at {r['question']!r}"
            e["ctx"] = torch.tensor(tok.encode(r["context_text"]).ids, dtype=torch.int32)
        assert all(pos[sp] == len(enc[sp]) for sp in enc), "encoded data has a different number of rows"
        for sp, rows in enc.items():
            torch.save(rows, os.path.join(args.patch_encoded, f"{sp}.pt"))
        print("added context to", args.patch_encoded, {sp: len(v) for sp, v in enc.items()})


if __name__ == "__main__":
    main()
