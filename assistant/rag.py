"""Retrieval-augmented answering for the server: transcribe the question, retrieve the closest answers, check.

Used when MODEL_DIR has a rag/ folder (finetune/package_assistant.py). Per question:
1. Whisper (already loaded for the audio encoder) transcribes it.
2. The fine-tuned retriever finds the top-k topics over the example wordings, for the question without the chatter
   around it (question_parts) and for its main part. A bare acknowledgement ("Of course.") gets no answer (ack).
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

# follow-up questions (rag.py, prepare): "and before that?" steps through the timeline; "what did you do there?"
# is searched together with the previous question
BEFORE = re.compile(r"\b(before (that|then|this|it)|earlier|previous(ly)?|prior to that)\b", re.I)
AFTER = re.compile(r"\b(after (that|this|it)|afterwards|then what|and then|next|later|what came next)\b", re.I)
# a bare "why?" / "how?" / "tell me more" is always about the previous answer, however well it matches on its own
BARE = re.compile(r"^\W*(and\s+|so\s+)?(why|how|how so|really|when|where|what else|tell me more|go on|in what way)\W*$", re.I)
REFERS = re.compile(r"\b(that|there|it|this|those|they|them|then)\b|^\W*(why|how|when|where|really|and)\W*$", re.I)
# an acknowledgement that needs no answer ("Of course.", "That's really impressive.", "Got it.")
ACK = re.compile(r"^\W*((ok(ay)?|right|sure|great|cool|nice|wow|yes|yeah|no|nope|exactly|indeed|perfect|understood|"
                 r"of course|got it|i see|i understand|that one|fair enough|makes sense|that makes( a lot of)? sense|"
                 r"(that'?s|that is|it'?s|how) (really |very |so )?(great|good|interesting|impressive|clear|helpful|"
                 r"fascinating|amazing|cool|nice|insightful|comprehensive|a great point))\W*)+$", re.I)
# chatty visitors: "That's really impressive. So what did you do at X? And how did Y?" -> the question parts only
ASKS = re.compile(r"\?|^\W*(and |so |but |okay |ok |now )*(what|how|why|when|where|who|whom|which|can|could|would|will|did|do|"
                  r"does|have|has|had|are|is|were|was|tell|describe|walk|explain|give|share|talk|any)\b", re.I)


def question_parts(text):
    """The parts of an utterance that ask something: leading acknowledgements and summaries are dropped, and a
    multi-part question ("What did you do at X? And how did it help Y?") is split, so each part can be searched."""
    sentences = [x.strip() for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x.strip()]
    asks = [x for x in sentences if ASKS.search(x)]
    parts = []
    for x in asks:
        parts += [y.strip(" ,") for y in re.split(r",? and (?=(?:what|how|why|when|where|who|which|did|do|does|have|has|were|was|are|is)\b)", x, flags=re.I) if y.strip(" ,")]
    return asks, parts

import numpy as np

from retrieval import Retriever


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


class RagFront:
    def __init__(self, rag_dir, tok, device="cuda", k=2, min_margin=0.02,
                 min_score=float(os.environ.get("RAG_MIN_SCORE", 0.85)),
                 rescue_score=float(os.environ.get("RAG_RESCUE_SCORE", 0.85))):
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
        # timeline.json: {"sequences": [[topic, topic, ...], ...]}, each list in time order (jobs, places, studies),
        # for "and before that?" / "what came next?"
        tl = os.path.join(rag_dir, "timeline.json")
        self.sequences = [[t for t in q if t in a] for q in json.load(open(tl, encoding="utf-8"))["sequences"]] \
            if os.path.exists(tl) else []
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

    def neighbour(self, topic, step):
        """The topic before (step -1) or after (+1) this one in a timeline, or None."""
        for q in self.sequences:
            if topic in q:
                i = q.index(topic) + step
                return q[i] if 0 <= i < len(q) else None
        return None

    def prepare(self, question_text, prev=None):
        """-> dict(topics, scores, margin, confident, ctx_ids, starts, followup) for AnswerSeq.
        prev: {"q": the previous question, "topic": its answer's topic}, for follow-up questions."""
        heard = question_text
        asks, parts = question_parts(question_text)
        if ACK.match(question_text):  # "Of course.", "That's really impressive.": nothing to answer
            found = self.retriever.search(question_text, k=self.k)
            if found[0][1] < self.min_score:  # (but "Thanks!" or "Hello" are small-talk topics with an answer)
                return {"ack": True, "topics": [t for t, _ in found], "scores": [s for _, s in found], "margin": 0.0,
                        "confident": False, "followup": None, "focus": question_text}
        if asks:
            question_text = " ".join(asks)
        # the question without the chatter, and its first part alone ("What was your role at X?" of "What was your role
        # at X, and what did you learn?"): the better match wins. Later parts are details, which match vaguely.
        found = max((self.retriever.search(q, k=self.k) for q in [question_text] + parts[:1] if q),
                    key=lambda f: f[0][1])
        followup = None
        main = parts[0] if parts else question_text
        if prev and BARE.match(question_text):
            found, followup = self.retriever.search(prev["q"] + " " + question_text, k=self.k), "bare follow-up"
        elif prev and found[0][1] < self.min_score:  # weak on its own: maybe it refers to the previous question
            step = -1 if BEFORE.search(question_text) else 1 if AFTER.search(question_text) else 0
            near = self.neighbour(prev["topic"], step) if step else None
            if near:  # "and before that?": the previous topic's neighbour in time, as a sure match
                found = [(near, max(self.rescue_score, found[0][1]))] + [x for x in found if x[0] != near][: self.k - 1]
                followup = f"timeline {'before' if step < 0 else 'after'} {prev['topic']}"
            elif REFERS.search(main) or len(words(main)) <= 4:  # in the main question, not its later details
                both = self.retriever.search(prev["q"] + " " + question_text, k=self.k)
                if both[0][1] > found[0][1]:
                    found, followup = both, "with the previous question"
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
                "allowed_text": allowed, "rescue": rescue, "followup": followup,
                "focus": question_text if question_text != heard else None}

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
