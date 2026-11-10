"""Finds the Q&A answers closest to a question, for retrieval-augmented answering.

The fine-tuned model alone picks the right answer for 63% of unseen wordings; matching a question against the
~1,700 example wordings with a sentence-embedding model finds the right topic far more often. So the retrieved
answers are put in front of the question and the model is trained to read and speak the right one.

* Index: every example question wording, with its topic. A topic's score is the best of its wordings.
* Optional reranker: re-scores the best candidates by reading the question and each wording together, which fixes
  look-alikes such as "Did he work in a bank?" vs "Did he work in Germany?".
"""

import json

import torch


def _mean_pool(h, mask):
    return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)


class Retriever:
    def __init__(self, embed_model="intfloat/multilingual-e5-base", rerank_model=None, device="cpu"):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(embed_model)
        self.emb = AutoModel.from_pretrained(embed_model).eval().to(device)
        self.rr_tok = self.rr = None
        if rerank_model:
            self.rr_tok = AutoTokenizer.from_pretrained(rerank_model)
            self.rr = AutoModelForSequenceClassification.from_pretrained(rerank_model).eval().to(device)

    @torch.inference_mode()
    def embed(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            b = self.tok(["query: " + t for t in texts[i: i + batch]], padding=True, truncation=True, max_length=64,
                         return_tensors="pt").to(self.device)
            out.append(torch.nn.functional.normalize(_mean_pool(self.emb(**b).last_hidden_state, b["attention_mask"]), dim=-1))
        return torch.cat(out)

    def build(self, wordings):
        """wordings: [(question, topic_id)]."""
        self.questions = [q for q, _ in wordings]
        self.topics = [t for _, t in wordings]
        self.E = self.embed(self.questions)
        return self

    @torch.inference_mode()
    def search(self, question, k=2, candidates=20, exclude=None, q_emb=None):
        """Top-k topics as [(topic, score)]. exclude: a wording to leave out (building training data)."""
        q = q_emb if q_emb is not None else self.embed([question])[0]
        s = self.E @ q
        if exclude is not None:
            s = s.clone()
            for i, w in enumerate(self.questions):
                if w == exclude:
                    s[i] = -1
        idx = s.topk(min(candidates, len(s))).indices.tolist()
        if self.rr is not None:
            pairs = [[question, self.questions[i]] for i in idx]
            b = self.rr_tok(pairs, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            r = self.rr(**b).logits.view(-1).float()
            scored = sorted(zip(idx, r.tolist()), key=lambda x: -x[1])
        else:
            scored = [(i, float(s[i])) for i in idx]
        best = {}
        for i, sc in scored:  # a topic scores as its best wording
            best.setdefault(self.topics[i], sc)
        return sorted(best.items(), key=lambda x: -x[1])[:k]


def load_wordings(manifest):
    """Training wordings of a manifest (jsonl rows with question, id, split)."""
    seen, out = set(), []
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "train" and r["question"] not in seen:
            seen.add(r["question"])
            out.append((r["question"], r["id"]))
    return out
