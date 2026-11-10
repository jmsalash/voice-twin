"""Several visitors' answers generated together: one decoder pass advances every active answer by one step.

A 0.5B model on a small GPU spends each step mostly reading its weights, the same work for 1 row or 8, so
batching serves several visitors at nearly the speed of one. Answering one visitor at a time instead makes the
others wait for whole answers; taking turns step by step would slow every answer below real time.

* A fixed pool of slots (2 rows each: the speaking row and the writing row) with a preallocated KV cache.
* A new question is prefilled on its own, written into a free slot and joins the batch at the next step.
* Each row attends only to its own cache entries (mask by length), at its own position (RoPE per row).
* Per answer the logic is engine.Assistant.answer's: the same streaming and "please rephrase" check.
"""

import math
import queue
import threading
import time

import torch

from engine import (  # noqa: F401
    ANSWER_A, CHUNK, EOA, EOT, INPUT_A, INPUT_T, MAX_CONTEXT, N_AUDIO, PAD_A, PAD_T, SAMPLES_PER_FRAME, ANSWER_T,
    embed, empty_past, layershift, words,
)
from omni_lm import HEAD, N_GROUPS, N_HEAD, Q_PER_KV, rope


class AnswerSeq:
    """One visitor's answer as a state machine: prompt() once, then on_tokens() for every decoder output."""

    def __init__(self, eng, wav16, emit, stop, rag=None):
        """rag: RagFront.prepare() output (retrieved answers in front, extra checks), or None."""
        self.eng, self.wav, self.emit, self.stop, self.rag = eng, wav16, emit, stop, rag
        self.sent = 0  # audio samples sent

    @torch.inference_mode()
    def prompt(self):
        e, w = self.eng, self.eng.whisper_mod
        T = min(1500, int(len(self.wav) / 16000 * 1000 / 20) + 1)
        mel = w.log_mel_spectrogram(w.pad_or_trim(torch.from_numpy(self.wav))).to(e.device)
        feat = e.whisper.embed_audio(mel.unsqueeze(0))[0][:T]
        rows = []
        for last_audio in (ANSWER_A, PAD_A):  # row 0 speaks (A1A2), row 1 writes (A1T2)
            s = [[layershift(INPUT_A, i)] + [layershift(PAD_A, i)] * T + [layershift(EOA, i), layershift(last_audio, i)]
                 for i in range(N_AUDIO)]
            s.append([INPUT_T] + [PAD_T] * T + [EOT, ANSWER_T])
            rows.append(s)
        off = 0
        if self.rag:  # retrieved answers as a text segment in front (finetune/common.build_sequence's layout)
            ctx = self.rag["ctx_ids"]
            off = len(ctx) + 2
            for row in rows:
                for i in range(N_AUDIO):
                    row[i] = [layershift(PAD_A, i)] * off + row[i]
                row[N_AUDIO] = [INPUT_T] + ctx + [EOT] + row[N_AUDIO]
        ids = torch.tensor(rows, device=e.device)
        L = off + T + 3
        mask = torch.zeros(2, L, dtype=torch.bool, device=e.device)
        mask[:, off + 1: off + T + 1] = True
        a = torch.zeros(2, L, 896, device=e.device)
        a[:, off + 1: off + T + 1] = e.ad(feat.unsqueeze(0))[0]
        self.L = L
        self.layers = [[] for _ in range(8)]
        self.flush = e.flusher(self.layers)
        self.text_ids, self.own_ids = [], []
        self.text_end = self.own_end = self.decided = False
        self.said, self.held, self.step, self.t0, self.first_audio = "", [], 0, time.time(), None
        return embed(e.wte, ids, a, mask)

    def _audio(self, chunks):
        for c in chunks:
            self.first_audio = self.first_audio or time.time() - self.t0
            self.sent += len(c)
            self.emit("audio", c)

    def _rescue(self, reason):
        """The retriever is very sure of the topic but the model refused or its rows disagree: play that topic's
        recorded answer (rag.py, step 3b). Only before any audio was sent."""
        return self._play_recorded(self.rag["rescue"], reason, rescued=True)

    def _play_recorded(self, t, reason="", rescued=False):
        """Say topic t's recorded answer (the model chose it, or it was rescued), then stop generating."""
        e = self.eng
        clip, full = e.rag.clips[t], e.rag.answers[t]
        self.emit("text", full)
        first = time.time() - self.t0
        for k in range(0, len(clip), CHUNK * SAMPLES_PER_FRAME):
            self.emit("audio", clip[k: k + CHUNK * SAMPLES_PER_FRAME])
        self.emit("done", {"rephrase": False, "rescued": rescued, "recorded": True, "text": full, "reason": reason, "topic": t,
                           "candidate": e.decode(self.text_ids), "own": e.decode(self.own_ids),
                           "firstAudioMs": int(1000 * first), "totalMs": int(1000 * (time.time() - self.t0))})
        return None

    def _finish_rephrase(self, reason=""):
        e = self.eng
        if self.rag and self.rag.get("rescue") and self.sent == 0:
            return self._rescue(reason)
        rtext, raudio = e.next_rephrase()
        self.emit("text", rtext)
        first = time.time() - self.t0
        for k in range(0, len(raudio), CHUNK * SAMPLES_PER_FRAME):
            self.emit("audio", raudio[k: k + CHUNK * SAMPLES_PER_FRAME])
        self.emit("done", {"rephrase": True, "text": rtext, "firstAudioMs": int(1000 * first),
                           "candidate": e.decode(self.text_ids), "own": e.decode(self.own_ids), "reason": reason,
                           "totalMs": int(1000 * (time.time() - self.t0))})
        return None

    def _finish(self, voice_ended=False):
        e = self.eng
        if voice_ended and not self.text_end and not e.clip_fallback:
            # the voice stopped while the text goes on: say "please rephrase" rather than half an answer. (Splicing in
            # the recorded clip from the same point sounded scrambled: the two never line up exactly.)
            return self._finish_rephrase("voice stopped early")
        if voice_ended and self.rag and not self.text_end:
            # the voice stopped while the text goes on (e.g. just "Yes."): if the rows agree so far and it is one of the
            # retrieved answers, play the rest from that answer's recorded clip; otherwise ask to rephrase
            tw, ow = words(e.decode(self.text_ids)), words(e.decode(self.own_ids))
            m = min(len(tw), len(ow), e.agree_words)
            ok = m > 0 and tw[:m] == ow[:m] and e.rag.in_context(self.rag, tw, m)
            clip, full = e.rag.clip_for(self.rag, tw) if ok else (None, None)
            if clip is None:
                return self._finish_rephrase("voice stopped early, no matching recording")
            if clip is not None:
                self._audio(self.held + self.flush(True))
                self.held = []
                rest = clip[min(self.sent, len(clip)):]
                for k in range(0, len(rest), CHUNK * SAMPLES_PER_FRAME):
                    self._audio([rest[k: k + CHUNK * SAMPLES_PER_FRAME]])
                if len(full) > len(self.said):
                    self.emit("text", full[len(self.said):])
                self.emit("done", {"rephrase": False, "text": full, "fallback": True,
                                   "firstAudioMs": int(1000 * (self.first_audio or 0)), "totalMs": int(1000 * (time.time() - self.t0))})
                return None
        if not self.decided and words(e.decode(self.text_ids)) != words(e.decode(self.own_ids)):
            return self._finish_rephrase("short answer: the two rows differ")
        final = e.decode(self.text_ids)
        if len(final) > len(self.said):
            self.emit("text", final[len(self.said):])
        self._audio(self.held + self.flush(True))
        self.emit("done", {"rephrase": False, "text": final, "firstAudioMs": int(1000 * (self.first_audio or 0)),
                           "steps": self.step, "totalMs": int(1000 * (time.time() - self.t0))})
        return None

    def on_tokens(self, tok):
        """tok: [[8 ints] row 0, [8 ints] row 1]. Returns the next input ids [[8], [8]] (shifted), or None when done."""
        e = self.eng
        if self.stop():
            self.emit("done", {"stopped": True})
            return None
        audio = tok[0][:N_AUDIO]
        text = PAD_T if self.text_end else tok[1][N_AUDIO]
        own = tok[0][N_AUDIO]
        if (self.step > 0 and audio[-1] == EOA) or self.step >= MAX_CONTEXT - self.L - 1:
            return self._finish(voice_ended=audio[-1] == EOA)
        if text == EOT:
            self.text_end = True
        elif not self.text_end:
            self.text_ids.append(text)
        if not self.own_end:
            if own == EOT:
                self.own_end = True
            else:
                self.own_ids.append(own)
        for i in range(N_AUDIO):
            self.layers[i].append(audio[i])
        self.layers[7].append(text)
        if not self.decided and self.rag and e.recorded:
            # recorded answers: the model's first words say which answer (or fallback) it gives; that answer's
            # recording is played, so the voice never garbles or drifts into the other retrieved answer
            n = e.agree_words
            tw = words(e.decode(self.text_ids))
            if len(tw) > n or self.text_end:
                self.decided = True
                t = e.rag.answer_for(self.rag, tw, n)
                if t == "_redirect":
                    if self.rag.get("rescue"):
                        return self._rescue("the model refused a question the retriever is sure of")
                    return self._finish_rephrase("the model chose a fallback line")
                if t is None:
                    return self._finish_rephrase("not one of the retrieved answers")
                return self._play_recorded(t)
            self.step += 1  # not decided yet: keep generating, but no audio goes out
            return [[layershift(x, i) for i, x in enumerate(audio)] + [text],
                    [layershift(PAD_A, i) for i in range(N_AUDIO)] + [text]]
        if not self.decided:
            n = e.agree_words
            tw, ow = words(e.decode(self.text_ids)), words(e.decode(self.own_ids))
            if (len(tw) > n or self.text_end) and (len(ow) > n or self.own_end):
                self.decided = True
                if tw[:n] != ow[:n]:
                    return self._finish_rephrase("the two rows disagree")
                if self.rag and not e.rag.in_context(self.rag, tw, n):  # not one of the retrieved answers
                    return self._finish_rephrase("not one of the retrieved answers")
                if self.rag and self.rag.get("rescue") and e.rag.is_redirect(tw, n):  # refuses a clear CV question
                    return self._rescue("the model refused a question the retriever is sure of")
        if self.decided:
            now = e.decode(self.text_ids)
            if len(now) > len(self.said) and not now.endswith("�"):
                self.emit("text", now[len(self.said):])
                self.said = now
            self._audio(self.held + self.flush(False))
            self.held = []
        else:
            self.held += self.flush(False)
        self.step += 1
        return [[layershift(x, i) for i, x in enumerate(audio)] + [text],
                [layershift(PAD_A, i) for i in range(N_AUDIO)] + [text]]


class Batcher:
    """Runs AnswerSeqs together on the GPU, in a thread of its own."""

    def __init__(self, eng, slots=4, capacity=MAX_CONTEXT):
        self.eng, self.slots, self.cap = eng, slots, capacity
        R, dev = 2 * slots, eng.device
        n_layer = len(eng.dec.h)
        self.K = [torch.zeros(R, N_GROUPS, capacity, HEAD, device=dev) for _ in range(n_layer)]
        self.V = [torch.zeros(R, N_GROUPS, capacity, HEAD, device=dev) for _ in range(n_layer)]
        self.lens = torch.zeros(R, dtype=torch.long, device=dev)
        self.rows = torch.arange(R, device=dev)
        self.free = list(range(slots))
        self.active, self.next = {}, {}
        self.pending = queue.Queue()
        self.wake = threading.Event()
        self.step_ms = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def busy(self):
        return len(self.active)

    def waiting(self):
        return self.pending.qsize() if not self.free else 0

    def submit(self, seq):
        self.pending.put(seq)
        self.wake.set()

    @torch.inference_mode()
    def _admit(self):
        while self.free and not self.pending.empty():
            seq = self.pending.get_nowait()
            s = min(self.free)  # lowest first, so a step only has to cover the slots in use
            self.free.remove(s)
            x = seq.prompt()
            L = x.shape[1]
            tokens, *present = self.eng.dec(x, *[p.to(self.eng.device) for p in empty_past(2)])
            for i in range(len(self.K)):
                self.K[i][2 * s: 2 * s + 2, :, :L] = present[2 * i]
                self.V[i][2 * s: 2 * s + 2, :, :L] = present[2 * i + 1]
            self.lens[2 * s: 2 * s + 2] = L
            nxt = seq.on_tokens(tokens.tolist())
            if nxt is None:
                self.free.append(s)
            else:
                self.active[s], self.next[s] = seq, nxt

    @torch.inference_mode()
    def _step(self, ids):
        """One decoder step for the first R rows (the slots up to the last one in use). ids: (R, 8, 1)."""
        dec, R = self.eng.dec, ids.shape[0]
        lens, rows = self.lens[:R], self.rows[:R]
        x = embed(self.eng.wte, ids)  # (R, 1, 896)
        pos = lens.float()
        idx = (pos[:, None] * dec.theta[None, :]).repeat(1, 2)  # (R, hs)
        cos, sin = idx.cos()[:, None, None, :], idx.sin()[:, None, None, :]
        S = int(lens.max()) + 1
        allowed = (torch.arange(S, device=x.device)[None, :] <= lens[:, None])[:, None, None, :]  # (R,1,1,S)
        for i, block in enumerate(dec.h):
            att = block.attn
            qkv = att.attn(block.norm_1(x)).view(R, 1, N_GROUPS, Q_PER_KV + 2, HEAD).permute(0, 2, 3, 1, 4)
            q = rope(qkv[:, :, :Q_PER_KV].reshape(R, N_HEAD, 1, HEAD), cos, sin)
            k = rope(qkv[:, :, Q_PER_KV], cos, sin)  # (R, G, 1, hs)
            v = qkv[:, :, Q_PER_KV + 1]
            self.K[i][rows, :, lens] = k[:, :, 0]
            self.V[i][rows, :, lens] = v[:, :, 0]
            kk = self.K[i][:R, :, :S].unsqueeze(2).expand(R, N_GROUPS, Q_PER_KV, S, HEAD).reshape(R, N_HEAD, S, HEAD)
            vv = self.V[i][:R, :, :S].unsqueeze(2).expand(R, N_GROUPS, Q_PER_KV, S, HEAD).reshape(R, N_HEAD, S, HEAD)
            scores = (q @ kk.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD))
            y = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ vv
            x = x + att.proj(y.transpose(1, 2).reshape(R, 1, N_HEAD * HEAD))
            x = x + block.mlp(block.norm_2(x))
        self.lens[:R] += 1
        from omni_lm import AUDIO_VOCAB, TEXT_VOCAB

        logits = (dec.ln_f(x) @ dec.lm_head_t)[:, 0]
        text = logits[:, :TEXT_VOCAB].argmax(-1, keepdim=True)
        audio = logits[:, TEXT_VOCAB:].reshape(R, N_AUDIO, AUDIO_VOCAB).argmax(-1)
        return torch.cat((audio, text), dim=1)

    def _loop(self):
        with torch.inference_mode():  # per thread: the audio decoding in on_tokens runs here too
            self._run()

    def _run(self):
        while True:
            try:
                self._admit()
                if not self.active:
                    self.wake.wait(0.2)
                    self.wake.clear()
                    continue
                n = max(self.active) + 1  # slots 0..n-1 take part in this step
                ids = torch.zeros(2 * n, 8, 1, dtype=torch.long)
                for s in range(n):
                    if s in self.active:
                        ids[2 * s: 2 * s + 2, :, 0] = torch.tensor(self.next[s])
                    else:
                        self.lens[2 * s: 2 * s + 2] = 0  # idle slots: keep their (unused) rows at position 0
                t = time.time()
                tokens = self._step(ids.to(self.eng.device)).tolist()
                self.step_ms = 0.9 * self.step_ms + 100 * (time.time() - t)
                for s in list(self.active):
                    nxt = self.active[s].on_tokens(tokens[2 * s: 2 * s + 2])
                    if nxt is None:
                        del self.active[s], self.next[s]
                        self.free.append(s)
                    else:
                        self.next[s] = nxt
            except Exception as ex:  # keep serving: fail the answers in flight
                for s, seq in list(self.active.items()):
                    seq.emit("error", f"{type(ex).__name__}: {ex}")
                    seq.emit("done", {"stopped": True})
                    self.free.append(s)
                self.active.clear()
                self.next.clear()
