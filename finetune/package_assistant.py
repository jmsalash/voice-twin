"""Builds the folder the assistant server mounts (MODEL_DIR): the fine-tuned model plus everything RAG needs.

  python package_assistant.py --ckpt work/ckpt_v5 --out /path/to/assistant-model
      [--manifest work/cv/manifest.v4.jsonl] [--general work/general/manifest.v4.jsonl]
      [--retriever work/retriever] [--extra data/retrieval_extra.json] [--clips work/answers_v4]

  OUT/lit_model.pth, tokenizer.json            the fine-tuned Mini-Omni2
  OUT/rephrase*.wav, rephrase*.txt             the "please rephrase" lines in the assistant's voice (said in turn)
  OUT/disclaimer.wav, disclaimer.txt           first person: "I'm an AI version of ..." played when a conversation starts
  OUT/rag/retriever/                           the fine-tuned embedding model
  OUT/rag/index.json                           [[wording, topic]]: training wordings + extra text-only ones
  OUT/rag/answers.json                         {topic: answer}, and {"_redirects": [...]} for off-topic replies
  OUT/rag/clips/<topic>.wav                    recorded answers (played for the answer the model chooses)
  OUT/rag/timeline.json                        optional: topics in time order, for "and before that?" (--timeline)
"""

import argparse
import json
import os
import shutil

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=os.path.join(C.WORK, "cv", "manifest.v4.jsonl"))
    ap.add_argument("--general", default=os.path.join(C.WORK, "general", "manifest.v4.jsonl"))
    ap.add_argument("--retriever", default=os.path.join(C.WORK, "retriever"))
    ap.add_argument("--extra", default=os.path.join(HERE, "data", "retrieval_extra.json"))
    ap.add_argument("--clips", default=os.path.join(C.WORK, "answers_v4"))
    ap.add_argument("--fixed-lines", default="", help="folder with rephrase.wav/.txt and (first person) disclaimer.wav/.txt "
                    "in the assistant's voice; overrides the checkpoint's")
    ap.add_argument("--timeline", default=os.path.join(HERE, "data", "timeline.json"),
                    help='{"sequences": [[topic, ...], ...]}: topics in time order (jobs, studies, places), for follow-ups')
    ap.add_argument("--vocabulary", default=os.path.join(HERE, "data", "vocabulary.txt"),
                    help="names visitors will say (companies, products, places), one per line: a hint for Whisper")
    args = ap.parse_args()

    rag = os.path.join(args.out, "rag")
    os.makedirs(os.path.join(rag, "clips"), exist_ok=True)
    for f in ("lit_model.pth", "tokenizer.json", "tokenizer_config.json", "model_config.yaml", "rephrase.wav", "rephrase.txt"):
        src = os.path.join(args.ckpt, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    if args.fixed_lines:
        import glob
        lines = glob.glob(os.path.join(args.fixed_lines, "rephrase*.*")) + glob.glob(os.path.join(args.fixed_lines, "disclaimer.*"))
        if any(os.path.basename(f).startswith("rephrase") for f in lines):  # the new set replaces the old one
            for old in glob.glob(os.path.join(args.out, "rephrase*.*")):
                os.remove(old)
        for f in lines:
            shutil.copy(f, os.path.join(args.out, os.path.basename(f)))
    if os.path.isdir(os.path.join(rag, "retriever")):
        shutil.rmtree(os.path.join(rag, "retriever"))
    shutil.copytree(args.retriever, os.path.join(rag, "retriever"))

    rows = C.read_jsonl(args.manifest)
    answers = {r["id"]: r["answer"] for r in rows}
    seen, index = set(), []
    for r in rows:
        if r["split"] == "train" and r["question"] not in seen:
            seen.add(r["question"])
            index.append([r["question"], r["id"]])
    if args.extra and os.path.exists(args.extra):
        for t, qs in json.load(open(args.extra, encoding="utf-8")).items():
            if t.startswith("_"):
                continue
            if t not in answers:  # a topic that no longer exists would make the server fail on that question
                print(f"warning: {args.extra}: no topic {t!r}, its {len(qs)} wordings skipped")
                continue
            index += [[q, t] for q in qs if q not in seen]
    redirects = sorted({r["answer"] for r in C.read_jsonl(args.general)})
    json.dump(index, open(os.path.join(rag, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump({**answers, "_redirects": redirects}, open(os.path.join(rag, "answers.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if args.vocabulary and os.path.exists(args.vocabulary):
        shutil.copy(args.vocabulary, os.path.join(rag, "vocabulary.txt"))
    if args.timeline and os.path.exists(args.timeline):
        tl = json.load(open(args.timeline, encoding="utf-8"))
        unknown = [t for q in tl["sequences"] for t in q if t not in answers]
        if unknown:
            print(f"warning: {args.timeline}: no topics {unknown}, left out of the timeline")
        json.dump({"sequences": [[t for t in q if t in answers] for q in tl["sequences"]]},
                  open(os.path.join(rag, "timeline.json"), "w", encoding="utf-8"), indent=1)
    n = 0
    for t in answers:
        clip = os.path.join(args.clips, t + ".wav")
        if os.path.exists(clip):
            shutil.copy(clip, os.path.join(rag, "clips", t + ".wav"))
            n += 1
    print(f"{args.out}: model, {len(index)} index wordings, {len(answers)} answers + {len(redirects)} redirects, {n} clips")


if __name__ == "__main__":
    main()
