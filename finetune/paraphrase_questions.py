"""Writes many more phrasings of each Q&A question with a local LLM (Ollama), so the fine-tuned model learns what a
question means instead of how a few exact questions sound.

With 3 to 5 phrasings per topic the model memorised them: trained wordings were answered 58/58, unseen wordings
only 10/58. The new phrasings are training-only; the held-out phrasing of each topic stays out (and anything close
to it is dropped), so the evaluation still uses wordings the model never heard.

  python paraphrase_questions.py [--n 25] [--model qwen3:4b-instruct] [--url http://127.0.0.1:11435]
  -> data/cv_qa.paraphrases.json  {id: [questions]}; used by 01_cv_audio.py --extra-questions
"""

import argparse
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPT_FIRST = """On a personal website, visitors talk to an AI double of {person}, which answers in the first person, as if
they were talking to {person}. They address it as "you". This is one of its answers:

"{answer}"

Visitors asked for it with questions such as:
{examples}

Write {n} more questions that a visitor might say out loud and that this answer answers well. Make them varied:
short and long, casual and formal, interview style ("tell me about...", "walk me through...", "how many years..."),
always addressing {person} as "you". Each question must ask for what this whole answer says, the way the examples
do: never for just one detail of it, which other answers cover. Plain spoken English.
Return only a JSON list of strings."""

PROMPT = """A voice assistant on a personal website answers visitors' spoken questions about {person}.
This is one of its answers:

"{answer}"

Visitors asked for it with questions such as:
{examples}

Write {n} more questions that a visitor might say out loud and that this answer answers well. Make them varied:
short and long, casual and formal, with and without the name ("he", "{first}", "this person", "the owner of this site"),
direct questions and requests such as "tell me about...". Each question must ask for what this whole answer says,
the way the examples do: never for just one detail of it, which other answers cover. Plain spoken English.
Return only a JSON list of strings."""


def norm(q):
    return re.sub(r"[^a-z0-9 ]+", "", q.lower()).split()


def close(a, b):
    """Same or nearly the same wording (most words shared)."""
    x, y = set(norm(a)), set(norm(b))
    return len(x & y) / max(1, len(x | y)) >= 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=os.path.join(HERE, "data", "cv_qa.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "cv_qa.paraphrases.json"))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="re-check the existing file and keep only questions the checker gives to their own topic")
    args = ap.parse_args()

    qa = json.load(open(args.qa, encoding="utf-8"))
    # the held-out phrasings, as 01_cv_audio.py chose them
    work = os.environ.get("FT_WORK", os.path.join(HERE, "work"))
    held_by_id = {}
    for l in open(os.path.join(work, "cv", "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == "eval":
            held_by_id[r["id"]] = r["question"]
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    menu = "\n".join(f"{it['id']}: {it['answer']}" for it in qa["items"])

    def ask(content, num_predict, temperature):
        body = {"model": args.model, "stream": False, "keep_alive": "10m",
                "options": {"temperature": temperature, "num_predict": num_predict, "seed": args.seed},
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(args.url + "/api/chat", json.dumps(body).encode(), {"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["message"]["content"]

    def pick(q):
        """Cross-check: which answer does the LLM choose for this question, out of all of them?"""
        a = ask(f"These are a voice assistant's answers, each with an id:\n\n{menu}\n\nA visitor asks: \"{q}\"\n"
                "Which single answer answers it best? Reply with the id only.", 20, 0)
        return a.strip().strip("`'\". ").split()[0] if a.strip() else ""

    if args.strict:
        # Accepting the aliases let in questions that belong to the other topic (who-questions under greeting):
        # the model then learns two answers for one kind of question and blends them.
        for it in qa["items"]:
            before = out.get(it["id"], [])
            out[it["id"]] = [q for q in before if pick(q) == it["id"]]
            print(f"{it['id']}: {len(out[it['id']])} of {len(before)} kept", flush=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("written", args.out, sum(len(v) for v in out.values()), "questions")
        return

    for it in qa["items"]:
        held = held_by_id[it["id"]]
        if it["id"] in out:
            continue
        shown = [q for q in it["questions"] if q != held]
        text = ask((PROMPT_FIRST if qa.get("mode") == "first_person" else PROMPT).format(person=qa["person"], first=qa["person"].split()[0], answer=it["answer"], n=args.n,
                                 examples="\n".join(f"- {q}" for q in shown)), 1500, 0.9)
        m = re.search(r"\[.*\]", text, re.S)
        try:
            cand = [q.strip() for q in json.loads(m.group(0)) if isinstance(q, str)] if m else []
        except json.JSONDecodeError:
            cand = []
        keep = []
        for q in cand:
            if 2 <= len(q.split()) <= 25 and not close(q, held) and not any(close(q, k) for k in keep + it["questions"]):
                keep.append(q)
        # answers the checker also gives for the topic's own (human-written) questions count as fine: overlapping
        # topics such as "who" and "summary" are both good replies to "Tell me about him"
        ok = {it["id"]} | {pick(q) for q in shown}
        right = [q for q in keep if pick(q) in ok]
        out[it["id"]] = right
        print(f"{it['id']}: {len(right)} kept of {len(keep)} (also accepted: {sorted(ok - {it['id']})}, held out: {held!r})", flush=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", args.out, sum(len(v) for v in out.values()), "questions")


if __name__ == "__main__":
    main()
