"""Precision vs. "please rephrase": how many right / wrong answers are spoken under different safety rules.

Reads rag_eval.py results. Signals per answer:
  agree     the two rows' texts start with the same words (the safety net)
  margin    retriever score of the best topic minus the second (confidence)
  in_ctx    the answer is one of the retrieved answers, or the redirect (it did not answer from elsewhere)
  short     the voice is much shorter than the text (an answer cut off after a word or two)
"right" means the answer is (word for word) the expected one or an accepted alternative; anything else counts as
not right, so incomplete answers count against a rule here (conservative).

  python precision_curve.py work/ckpt_v5/eval_rag_e2e/results.jsonl
"""

import argparse
import json
import os

import common as C
from score_eval import wer, words

REDIRECT = ("only able", "only answer", "outside what")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--answers", default=os.path.join(C.WORK, "cv", "manifest.v4.jsonl"))
    ap.add_argument("--acceptable", default=os.path.join(C.HERE, "data", "eval_acceptable.json"))
    ap.add_argument("--agree-words", type=int, default=6)
    args = ap.parse_args()
    ans = {r["id"]: r["answer"] for r in C.read_jsonl(args.answers)}
    alt = json.load(open(args.acceptable, encoding="utf-8"))
    rows = C.read_jsonl(args.results)
    for r in rows:
        r["redirect"] = any(x in r["model"] for x in REDIRECT)
        cands = [r["expected"]] + [ans[t] for t in alt.get(r["question"], [])]
        r["right"] = min(wer(a, r["model"]) for a in cands) <= 0.1 if r["kind"] == "cv" else r["redirect"]
        n = args.agree_words
        r["agree"] = words(r["model"])[:n] == words(r["own"])[:n]
        r["margin"] = r["scores"][0] - r["scores"][1] if len(r["scores"]) > 1 else 1.0
        r["in_ctx"] = r["redirect"] or any(wer(ans[t], r["model"]) <= 0.1 for t in r["topics"])
        r["short"] = r["frames"] < 0.6 * 4.8 * len(words(r["model"]))  # ~4.8 SNAC frames per spoken word
    cv = [r for r in rows if r["kind"] == "cv"]
    off = [r for r in rows if r["kind"] != "cv"]
    print(f"{len(cv)} CV questions, {len(off)} off-topic; with no rule: {sum(r['right'] for r in cv)} right, "
          f"{sum(not r['right'] for r in cv)} not right; voice cut short in {sum(r['short'] for r in cv)}")

    def rule(name, ok):
        spoken = [r for r in cv if ok(r)]
        right = sum(r["right"] for r in spoken)
        wrong = len(spoken) - right
        off_bad = sum(1 for r in off if ok(r) and not r["right"])
        print(f"{name:52s} spoken {len(spoken):2d}: right {right:2d}, not right {wrong:2d} "
              f"(precision {100 * right / max(1, len(spoken)):5.1f}%) | rephrase {len(cv) - len(spoken):2d} "
              f"| off-topic answered wrongly {off_bad}")

    rule("no rule", lambda r: True)
    rule("safety net (rows agree)", lambda r: r["agree"])
    rule("answer is a retrieved answer", lambda r: r["in_ctx"])
    rule("safety net + retrieved answer", lambda r: r["agree"] and r["in_ctx"])
    rule("safety net + retrieved + voice not cut short", lambda r: r["agree"] and r["in_ctx"] and not r["short"])
    for m in (0.02, 0.05, 0.1):
        rule(f"... + retriever margin >= {m}", lambda r, m=m: r["agree"] and r["in_ctx"] and not r["short"] and r["margin"] >= m)
    print("\nnot right but spoken under 'safety net + retrieved + not cut short':")
    for r in cv:
        if r["agree"] and r["in_ctx"] and not r["short"] and not r["right"]:
            print(f"  {r['question']!r} (heard {r['heard']!r}) -> {r['model'][:90]!r}  topics {r['topics']} margin {r['margin']:.3f}")


if __name__ == "__main__":
    main()
