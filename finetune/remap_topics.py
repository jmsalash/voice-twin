"""Merges overlapping Q&A topics without re-recording any question.

Topics that compete for the same questions (msc / italy / phd / education...) make the model pick a neighbour's
answer. This maps each old topic to its merged one, so every existing question clip (original or paraphrased)
points to the merged answer, and lists the answers that need new audio.

  python remap_topics.py --merge data/cv_qa.v4.merge.json [--pronounce Siobhan=Shivawn]
  -> data/cv_qa.v4.json, work/{cv,general}/manifest.v4.jsonl, work/answers_v4.jobs.json
     (speak the jobs with: omnivoice_data.py --ref REF --jobs work/answers_v4.jobs.json)

--pronounce only changes the text given to the TTS (how a name should sound); the trained text keeps the spelling.
"""

import argparse
import json
import os

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=os.path.join(C.HERE, "data", "cv_qa.json"))
    ap.add_argument("--merge", required=True)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--pronounce", action="append", default=[], help="WRITTEN=SPOKEN, for the TTS only")
    args = ap.parse_args()

    qa = json.load(open(args.qa, encoding="utf-8"))
    merge = json.load(open(args.merge, encoding="utf-8"))
    new_id = {old: new for new, m in merge.items() for old in m["from"]}
    items, answer = [], {}
    for it in qa["items"]:
        nid = new_id.get(it["id"], it["id"])
        if nid not in answer:
            answer[nid] = merge[nid]["answer"] if nid in merge else it["answer"]
            items.append({"id": nid, "questions": [], "answer": answer[nid], "unknown": it.get("unknown", False)})
        next(x for x in items if x["id"] == nid)["questions"] += it["questions"]
    json.dump({**qa, "items": items}, open(os.path.join(C.HERE, "data", f"cv_qa.{args.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def say(text):
        for p in args.pronounce:
            written, spoken = p.split("=", 1)
            text = text.replace(written, spoken)
        return text

    out_dir = os.path.join(C.WORK, f"answers_{args.tag}")
    jobs = {}
    for kind in ("cv", "general"):
        rows = C.read_jsonl(os.path.join(C.WORK, kind, "manifest.ov.jsonl"))
        for r in rows:
            if kind == "cv":
                r["id"] = new_id.get(r["id"], r["id"])
                r["answer"] = answer[r["id"]]
                key = r["id"]
            else:
                key = os.path.splitext(os.path.basename(r["answer_wav"]))[0]  # offtopic_0..2
            r["answer_wav"] = os.path.join(out_dir, key + ".wav")
            jobs[r["answer_wav"]] = say(r["answer"])
        C.write_jsonl(os.path.join(C.WORK, kind, f"manifest.{args.tag}.jsonl"), rows)
        print(f"{kind}: {len(rows)} examples, {len({r['answer_wav'] for r in rows})} answers")
    os.makedirs(out_dir, exist_ok=True)
    json.dump([{"text": t, "path": p} for p, t in jobs.items()], open(os.path.join(C.WORK, f"answers_{args.tag}.jobs.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} topics, {len(jobs)} answers to speak -> {out_dir}")


if __name__ == "__main__":
    main()
