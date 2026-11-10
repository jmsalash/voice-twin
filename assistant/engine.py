"""A fine-tuned Mini-Omni2 answering spoken questions, with streamed audio and a "please rephrase" safety net.

Mini-Omni2 answers with two rows over the same question (as mini-omni2-web's export/pipeline.py): row 1 writes the text, row 0 speaks
it. Row 0 also predicts text of its own, which is normally thrown away. When the two rows disagree on the first
words, the answer is usually wrong (and the voice, made to follow the other row's text, garbles). So no audio is
sent until both rows have AGREE_WORDS words; if they differ, the visitor hears a "could you rephrase" line instead.
On the held-out questions of one fine-tune this turned 32 wrong answers out of 64 into 14, blocking 3 right ones.
"""

import os
import re
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "model"), os.path.join(HERE, "model")):
    sys.path.insert(0, p)

from omni_lm import (  # noqa: E402
    ANSWER_A, ANSWER_T, EOA, EOT, INPUT_A, INPUT_T, N_AUDIO, PAD_A, PAD_T, embed, empty_past, layershift, load,
)

TEXT_VOCAB_REAL = 151936  # ids above are special tokens
CHUNK, LEFT, LOOK = 4, 8, 1  # SNAC frames per audio chunk, left context, lookahead
SAMPLES_PER_FRAME = 2048
MAX_CONTEXT = 2048


class Stopped(Exception):
    pass


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


class Assistant:
    def __init__(self, model_dir, whisper_path, device="cuda", agree_words=6, rag_k=2, min_margin=0.02,
                 rephrase="Sorry, I didn't quite get that. Could you ask it another way?"):
        import whisper
        from snac import SNAC
        from tokenizers import Tokenizer

        t = time.time()
        self.device, self.agree_words = device, agree_words
        dec, ad, wte = load(os.path.join(model_dir, "lit_model.pth"))  # wte is a plain tensor
        self.dec, self.ad, self.wte = dec.to(device).eval(), ad.to(device).eval(), wte.to(device)
        self.whisper_mod = whisper
        self.whisper = whisper.load_model(whisper_path, device=device).eval()
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.clip_fallback = os.environ.get("CLIP_FALLBACK", "0") == "1"  # off: splicing recordings sounded scrambled
        # with RAG: play the recorded answer the model chose instead of its generated voice (never garbles)
        self.recorded = os.environ.get("RECORDED_ANSWERS", "1") == "1"
        self.disclaimer_audio, self.disclaimer_text = None, ""  # first person: played when a conversation starts
        clip = os.path.join(model_dir, "disclaimer.wav")
        if os.path.exists(clip):
            import soundfile as sf

            self.disclaimer_audio, _ = sf.read(clip, dtype="float32")
            txt = os.path.splitext(clip)[0] + ".txt"
            self.disclaimer_text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
        self.rag = None  # retrieval-augmented answering, when the model folder has rag/ (finetune/package_assistant.py)
        if os.path.isdir(os.path.join(model_dir, "rag")):
            from rag import RagFront

            self.rag = RagFront(os.path.join(model_dir, "rag"), self.tok, device, k=rag_k, min_margin=min_margin)
        self.gpu = threading.Lock()  # one generation at a time
        # The "please rephrase" lines: recordings in the assistant's voice (rephrase.wav, rephrase_1.wav, ... each with
        # its .txt in the model folder), said in turn so it doesn't repeat itself. Without any, the model speaks the
        # default line, but a fine-tuned model garbles text it never trained on.
        import glob
        import soundfile as sf

        self.rephrases, self._next_rephrase = [], 0
        for clip in sorted(glob.glob(os.path.join(model_dir, "rephrase*.wav"))):
            audio, sr = sf.read(clip, dtype="float32")
            assert sr == 24000, f"{clip} must be 24 kHz"
            txt = os.path.splitext(clip)[0] + ".txt"
            self.rephrases.append((open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else rephrase, audio))
        if not self.rephrases:
            with torch.inference_mode():
                self.rephrases.append((rephrase, np.concatenate([a for kind, a in self.say(rephrase) if kind == "audio"])))
        for _ in self.answer(np.zeros(16000, np.float32)):  # warm-up: the first answer pays CUDA's start-up (~8 s)
            pass
        if self.rag is not None:  # and the first transcription and retrieval theirs
            self.rag.prepare(self.rag.transcribe(self.whisper, np.zeros(16000, np.float32)) or "Hello")
        self.load_seconds = round(time.time() - t, 1)

    # ----- helpers -----

    def decode(self, ids):
        return self.tok.decode([t for t in ids if t < TEXT_VOCAB_REAL]).strip()

    def snac_window(self, layers, a, b):
        def code(layer, idx):
            return min(layers[layer][idx], 4095)

        c0, c1, c2 = [], [], []
        for f in range(a, b):
            c0.append(code(0, f + 1))
            c1 += [code(1, f + 2), code(4, f + 5)]
            c2 += [code(2, f + 3), code(3, f + 4), code(5, f + 6), code(6, f + 7)]
        return [torch.tensor([c], device=self.device) for c in (c0, c1, c2)]

    def flusher(self, layers):
        """Returns flush(final) -> list of float32 chunks for the frames now complete (with the delay undone)."""
        state = {"emitted": 0}

        def flush(final):
            out = []
            available = len(layers[6]) - 7
            while available - state["emitted"] >= (1 if final else CHUNK + LOOK):
                e0 = state["emitted"]
                k = min(CHUNK, available - e0)
                look = min(LOOK, available - e0 - k)
                s, e = max(0, e0 - LEFT), e0 + k + look
                wav = self.snac.decode(self.snac_window(layers, s, e)).squeeze().float().cpu().numpy()
                off = (e0 - s) * SAMPLES_PER_FRAME
                out.append(wav[off: off + k * SAMPLES_PER_FRAME])
                state["emitted"] += k
            return out

        return flush

    # ----- answering a spoken question -----

    @torch.inference_mode()
    def answer(self, wav16, stop=lambda: False):
        """wav16: float32 16 kHz mono. Yields ("text", piece), ("audio", float32 24 kHz), then
        ("done", {"rephrase": bool, "text": str, ...})."""
        w = self.whisper_mod
        T = min(1500, int(len(wav16) / 16000 * 1000 / 20) + 1)
        mel = w.log_mel_spectrogram(w.pad_or_trim(torch.from_numpy(wav16))).to(self.device)
        feat = self.whisper.embed_audio(mel.unsqueeze(0))[0][:T]
        rows = []
        for last_audio in (ANSWER_A, PAD_A):  # row 0 speaks (A1A2), row 1 writes (A1T2)
            s = [[layershift(INPUT_A, i)] + [layershift(PAD_A, i)] * T + [layershift(EOA, i), layershift(last_audio, i)]
                 for i in range(N_AUDIO)]
            s.append([INPUT_T] + [PAD_T] * T + [EOT, ANSWER_T])
            rows.append(s)
        ids = torch.tensor(rows, device=self.device)
        mask = torch.zeros(2, T + 3, dtype=torch.bool, device=self.device)
        mask[:, 1: T + 1] = True
        a = torch.zeros(2, T + 3, 896, device=self.device)
        a[:, 1: T + 1] = self.ad(feat.unsqueeze(0))[0]
        tokens, *past = self.dec(embed(self.wte, ids, a, mask), *[p.to(self.device) for p in empty_past(2)])

        layers = [[] for _ in range(8)]
        flush = self.flusher(layers)
        text_ids, own_ids = [], []
        text_end = own_end = False
        decided, said, held = False, "", []
        t0 = time.time()
        first_audio = None
        for step in range(MAX_CONTEXT - T - 3):
            if stop():
                raise Stopped
            audio = tokens[0, :N_AUDIO].tolist()
            text = PAD_T if text_end else int(tokens[1, N_AUDIO])
            own = int(tokens[0, N_AUDIO])
            if step > 0 and audio[-1] == EOA:
                break
            if text == EOT:
                text_end = True
            elif not text_end:
                text_ids.append(text)
            if not own_end:
                if own == EOT:
                    own_end = True
                else:
                    own_ids.append(own)
            for i in range(N_AUDIO):
                layers[i].append(audio[i])
            layers[7].append(text)

            if not decided:
                n = self.agree_words
                tw, ow = words(self.decode(text_ids)), words(self.decode(own_ids))
                if (len(tw) > n or text_end) and (len(ow) > n or own_end):
                    decided = True
                    if tw[:n] != ow[:n]:  # the two rows would give different answers
                        yield from self._rephrase(t0, self.decode(text_ids), self.decode(own_ids))
                        return
            if decided:
                now = self.decode(text_ids)
                if len(now) > len(said) and not now.endswith("�"):
                    yield "text", now[len(said):]
                    said = now
                for chunk in held + flush(False):
                    first_audio = first_audio or time.time() - t0
                    yield "audio", chunk
                held = []
            else:
                held += flush(False)  # keep the audio until the rows agree
            nxt = torch.tensor([[layershift(x, i) for i, x in enumerate(audio)] + [text],
                                [layershift(PAD_A, i) for i in range(N_AUDIO)] + [text]], device=self.device).unsqueeze(-1)
            tokens, *past = self.dec(embed(self.wte, nxt), *past)
        if not decided:  # very short answer: compare what there is
            if words(self.decode(text_ids)) != words(self.decode(own_ids)):
                yield from self._rephrase(t0, self.decode(text_ids), self.decode(own_ids))
                return
        final = self.decode(text_ids)
        if len(final) > len(said):
            yield "text", final[len(said):]
        for chunk in held + flush(True):
            first_audio = first_audio or time.time() - t0
            yield "audio", chunk
        yield "done", {"rephrase": False, "text": final, "firstAudioMs": int(1000 * (first_audio or 0)),
                       "steps": step, "totalMs": int(1000 * (time.time() - t0))}

    def next_rephrase(self):
        """The next "please rephrase" line, (text, audio), in turn."""
        r = self.rephrases[self._next_rephrase % len(self.rephrases)]
        self._next_rephrase += 1
        return r

    def _rephrase(self, t0, text, own):
        rtext, raudio = self.next_rephrase()
        yield "text", rtext
        first = time.time() - t0
        for k in range(0, len(raudio), CHUNK * SAMPLES_PER_FRAME):
            yield "audio", raudio[k: k + CHUNK * SAMPLES_PER_FRAME]
        yield "done", {"rephrase": True, "text": rtext, "firstAudioMs": int(1000 * first),
                       "candidate": text, "other": own, "totalMs": int(1000 * (time.time() - t0))}

    # ----- speaking a fixed text (for the rephrase line) -----

    def say(self, text):
        """Speaks text word for word in the fine-tuned voice (typed-input layout, the text stream forced)."""
        ids = self.tok.encode(text).ids
        n = len(ids)
        streams = [[layershift(PAD_A, i)] * (n + 2) + [layershift(ANSWER_A, i)] for i in range(N_AUDIO)]
        streams.append([INPUT_T] + ids + [EOT, ANSWER_T])
        tokens, *past = self.dec(embed(self.wte, torch.tensor([streams], device=self.device)),
                                 *[p.to(self.device) for p in empty_past(1)])
        forced = ids + [EOT]
        layers = [[] for _ in range(8)]
        flush = self.flusher(layers)
        for k in range(MAX_CONTEXT - n - 3):
            audio = tokens[0, :N_AUDIO].tolist()
            if k > 0 and audio[-1] == EOA:
                break
            t = forced[k] if k < len(forced) else PAD_T
            for i in range(N_AUDIO):
                layers[i].append(audio[i])
            layers[7].append(t)
            for c in flush(False):
                yield "audio", c
            nxt = torch.tensor([[[layershift(x, i)] for i, x in enumerate(audio)] + [[t]]], device=self.device)
            tokens, *past = self.dec(embed(self.wte, nxt), *past)
        for c in flush(True):
            yield "audio", c
