"""Retrieval-augmented answering for the server: transcribe the question, retrieve the closest answers, check.

Used when MODEL_DIR has a rag/ folder (finetune/package_assistant.py). Per question:
1. Whisper (already loaded for the audio encoder) transcribes it.
2. The fine-tuned retriever finds the top-k topics over the example wordings.
3. The k answers go in front of the question (training layout) and the model answers. The retrieval is trusted when
   the best wording is close to the question (score >= MIN_SCORE: off-topic questions match nothing well) or clearly
   ahead of the second (margin >= MIN_MARGIN); otherwise only a redirect is accepted and any other answer becomes
   "please rephrase".
3b. Rescue: when the best wording is very close (score >= RESCUE_SCORE) but the model refuses (says a redirect) or
   its two rows disagree, the best topic's recorded answer is played instead of "please rephrase".
4. Before any audio is sent, the answer must start like one of the retrieved answers or a redirect (in_context).
5. If the generated voice stops while the text goes on, the rest is played from that answer's recorded clip.
"""

import json
import os
import re

import numpy as np

from retrieval import Retriever


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


class RagFront:
    def __init__(self, rag_dir, tok, device="cuda", k=2, min_margin=0.02,
                 min_score=float(os.environ.get("RAG_MIN_SCORE", 0.85)),
                 rescue_score=float(os.environ.get("RAG_RESCUE_SCORE", 0.90))):
        import soundfile as sf

        import threading

        self.k, self.min_margin, self.tok = k, min_margin, tok
        self.min_score, self.rescue_score = min_score, rescue_score
        self.asr_lock = threading.Lock()
        index = json.load(open(os.path.join(rag_dir, "index.json"), encoding="utf-8"))
        a = json.load(open(os.path.join(rag_dir, "answers.json"), encoding="utf-8"))
        self.redirects = a.pop("_redirects", [])
        index = [x for x in index if x[1] in a]  # wordings of a topic with no answer would fail at answer time
        self.answers = a
        self.retriever = Retriever(os.path.join(rag_dir, "retriever"), device=device).build([tuple(x) for x in index])
        self.clips = {}
        for t in a:
            p = os.path.join(rag_dir, "clips", t + ".wav")
            if os.path.exists(p):
                x, sr = sf.read(p, dtype="float32")
                if sr == 24000:
                    self.clips[t] = x
        self.n_index = len(index)
        vocab = os.path.join(rag_dir, "vocabulary.txt")  # names, one per line (package_assistant.py --vocabulary)
        names = [l.strip() for l in open(vocab, encoding="utf-8") if l.strip()] if os.path.exists(vocab) else []
        self.vocabulary = ("Questions about " + ", ".join(names) + ".") if names else None
        # ASR_MODEL: a faster-whisper model folder (e.g. large-v3-turbo) for the transcript, which hears short
        # questions better than the model's own Whisper small; 8-bit weights, so it is fast on older GPUs too
        self.asr = None
        if os.environ.get("ASR_MODEL"):
            from faster_whisper import WhisperModel

            self.asr = WhisperModel(os.environ["ASR_MODEL"], device="cuda" if device.startswith("cuda") else "cpu",
                                    compute_type=os.environ.get("ASR_COMPUTE", "int8"))

    def transcribe(self, whisper_model, wav16):
        with self.asr_lock:  # one transcription at a time: concurrent calls on one Whisper model can hang
            return self._transcribe(whisper_model, wav16)

    def _transcribe(self, whisper_model, wav16):
        # The names visitors will say, as a hint: Whisper otherwise misspells company and product names (e.g. "ACME ONE" as "Acme 1").
        if self.asr is not None:
            segs, _ = self.asr.transcribe(wav16, language="en", beam_size=1, initial_prompt=self.vocabulary,
                                          condition_on_previous_text=False, without_timestamps=True, vad_filter=False)
            return " ".join(s.text for s in segs).strip()
        r = whisper_model.transcribe(wav16, language="en", fp16=False, temperature=0.0, initial_prompt=self.vocabulary,
                                     condition_on_previous_text=False, without_timestamps=True)
        return r["text"].strip()

    def prepare(self, question_text):
        """-> dict(topics, scores, margin, confident, ctx_ids, starts) for AnswerSeq."""
        found = self.retriever.search(question_text, k=self.k)
        topics = [t for t, _ in found]
        scores = [s for _, s in found]
        margin = scores[0] - scores[1] if len(scores) > 1 else 1.0
        ctx = "\n".join(self.answers[t] for t in topics)
        confident = scores[0] >= self.min_score or margin >= self.min_margin
        # not confident: off-topic questions match nothing well, so the model still answers, but only its redirect is
        # let through; any other answer becomes "please rephrase"
        allowed = ([self.answers[t] for t in topics] if confident else []) + self.redirects
        rescue = topics[0] if scores[0] >= self.rescue_score and topics[0] in self.clips else None
        return {"topics": topics, "scores": scores, "margin": margin, "confident": confident,
                "ctx_ids": self.tok.encode(ctx).ids, "allowed": [words(x) for x in allowed],
                "allowed_text": allowed, "rescue": rescue}

    def answer_for(self, rag, text_words, n):
        """Which allowed answer the text starts like: a topic, "_redirect", or None."""
        allowed = rag["topics"] if rag["confident"] else []
        for t in allowed:
            if words(self.answers[t])[:n] == text_words[:n] and t in self.clips:
                return t
        return "_redirect" if self.is_redirect(text_words, n) else None

    def is_redirect(self, text_words, n):
        """The answer starts like one of the off-topic redirects."""
        return any(words(r)[:n] == text_words[:n] for r in self.redirects)

    @staticmethod
    def in_context(rag, text_words, n):
        """The answer starts like one of the retrieved answers (or a redirect)."""
        return any(w[:n] == text_words[:n] for w in rag["allowed"])

    def clip_for(self, rag, text_words):
        """The recorded clip of the retrieved answer this text is the start of, and that answer's text."""
        for t in rag["topics"]:
            w = words(self.answers[t])
            if len(text_words) and w[: len(text_words)] == text_words and t in self.clips:
                return self.clips[t], self.answers[t]
        return None, None
