"""Judges 05_eval.py answers by meaning, not wording, with a local LLM (Ollama), so models that rephrase (the
original Mini-Omni2) can be compared with ones that say the curated answers word for word.

For a CV question the judge sees every curated answer (the facts), the expected answer (and accepted alternatives)
and the model's answer, and labels it:
  right       answers the question and says nothing false
  incomplete  says nothing false but does not really answer (vague, cut off, a related but different answer)
  wrong       false facts, the wrong answer, or garbled
For an off-topic question it checks whether the answer politely declines (redirect) or not.

  python judge_eval.py RESULTS.jsonl [--answers work/cv/manifest.v4.jsonl] [--acceptable data/eval_acceptable.json]
"""

import argparse
import collections
import json
import os
import re
import urllib.request

import common as C

CV_PROMPT = """You check a voice assistant that answers questions about one person, using only these facts:

{facts}

Question: "{question}"
A correct reply: "{expected}"{alts}
The assistant's reply: "{answer}"

Label the assistant's reply with one word:
right: it answers the question, and everything it says agrees with the facts (other wording is fine).
incomplete: nothing in it is false, but it does not really answer this question (too vague, cut off, or about
something else).
wrong: it says something false or not in the facts, gives an answer to a different question as if it answered this
one, or is garbled.
Reply with the one word only."""

OFF_PROMPT = """A voice assistant on a personal website may only talk about the person's career. A visitor asked
something else: "{question}"
The assistant replied: "{answer}"
Did it politely decline and steer back to the person's career, without answering the request? Reply yes or no."""


def ask(url, model, prompt):
    body = {"model": model, "stream": False, "options": {"temperature": 0, "num_predict": 8, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--answers", default=os.path.join(C.WORK, "cv", "manifest.v4.jsonl"))
    ap.add_argument("--acceptable", default=os.path.join(C.HERE, "data", "eval_acceptable.json"))
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    args = ap.parse_args()
    answer_of = {r["id"]: r["answer"] for r in C.read_jsonl(args.answers)}
    facts = "\n".join(f"- {a}" for a in dict.fromkeys(answer_of.values()))
    alt = json.load(open(args.acceptable, encoding="utf-8")) if os.path.exists(args.acceptable) else {}
    rows = C.read_jsonl(args.results)
    count = collections.Counter()
    for r in rows:
        if r["kind"] == "cv":
            alts = "".join(f'\nAlso correct: "{answer_of[t]}"' for t in alt.get(r["question"], []))
            out = ask(args.url, args.model, CV_PROMPT.format(facts=facts, question=r["question"], expected=r["expected"],
                                                             alts=alts, answer=r["model"] or "(nothing)"))
            label = next((w for w in ("incomplete", "right", "wrong") if w in out), "wrong")
        else:
            out = ask(args.url, args.model, OFF_PROMPT.format(question=r["question"], answer=r["model"] or "(nothing)"))
            label = "redirect" if out.startswith("yes") else "answered"
        r["judge"] = label
        count[(r["kind"], label)] += 1
    cv = sum(v for (k, _), v in count.items() if k == "cv")
    off = sum(v for (k, _), v in count.items() if k != "cv")
    print(f"CV ({cv}): right {count[('cv', 'right')]}, incomplete {count[('cv', 'incomplete')]}, wrong {count[('cv', 'wrong')]}"
          f" | off-topic ({off}): redirected {count[('offtopic', 'redirect')]}, answered {count[('offtopic', 'answered')]}")
    C.write_jsonl(os.path.splitext(args.results)[0] + ".judged.jsonl", rows)


if __name__ == "__main__":
    main()
