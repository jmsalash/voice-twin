"""Shared pieces of the Mini-Omni2 fine-tuning pipeline (see README.md).

Training sequences follow exactly what the model sees at inference (mini-omni2-web: export/pipeline.py, src/omni/engine.js):

  prompt (L = T + 3):  audio streams [input_a, <whisper features> x T, eoa, answer_a | pad_a]
                       text stream   [input_t, pad_t x T, eot, answer_t]
  answer (S = N + 8 steps for N SNAC frames):
    audio stream i:  pad_a x (i + 1), the N codes of layer i, eoa, pad_a ... (delay pattern)
    text stream:     answer tokens, eot, pad_t ... (pad_t is forced at inference, so it carries no loss)

Row "A1A2" (answer_a) learns audio + text; row "A1T2" (pad_a, audio inputs stay pad) learns text only,
because at inference the text comes from the A1T2 row and is fed back into both rows.
"""

import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))

from omni_lm import (  # noqa: E402,F401
    ANSWER_A, ANSWER_T, AUDIO_VOCAB, EOA, EOT, INPUT_A, INPUT_T, N_AUDIO, PAD_A, PAD_T, TEXT_VOCAB, layershift,
)

CKPT_DIR = os.environ.get("MINI_OMNI2_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "mini-omni2"))
WORK = os.environ.get("FT_WORK", os.path.join(HERE, "work"))
IGNORE = -100


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- audio ----------

def load_audio_16k(path_or_bytes):
    import librosa
    import soundfile as sf

    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    return librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav


def frames_from_snac(codes):
    """SNAC codes [c0 (F), c1 (2F), c2 (4F)] -> (N, 7) frames in Mini-Omni2's stream order."""
    c0, c1, c2 = [c.reshape(-1).tolist() for c in codes]
    n = min(len(c0), len(c1) // 2, len(c2) // 4)
    return np.array([[c0[f], c1[2 * f], c2[4 * f], c2[4 * f + 1], c1[2 * f + 1], c2[4 * f + 2], c2[4 * f + 3]] for f in range(n)], np.int16)


def frames_from_answer_snac(s):
    """VoiceAssistant-400K's answer_snac string ('# a b c d e f g # ...') -> (N, 7) frames."""
    out = []
    for chunk in s.split("#"):
        v = chunk.split()
        if len(v) == 7:
            out.append([int(x) for x in v])
    return np.array(out, np.int16)


class SnacCodec:
    def __init__(self, device="cpu"):
        from snac import SNAC

        self.device = device
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    @torch.inference_mode()
    def encode(self, wav24):
        x = torch.from_numpy(np.asarray(wav24, np.float32)).view(1, 1, -1).to(self.device)
        return frames_from_snac([c.cpu() for c in self.model.encode(x)])

    @torch.inference_mode()
    def decode(self, frames):
        f = torch.as_tensor(np.asarray(frames, np.int64))
        codes = [f[:, 0].view(1, -1), f[:, [1, 4]].reshape(1, -1), f[:, [2, 3, 5, 6]].reshape(1, -1)]
        return self.model.decode([c.to(self.device) for c in codes]).squeeze().cpu().numpy()


class WhisperFeatures:
    """openai-whisper small encoder, exactly as Mini-Omni2 uses it (30 s padded mel, features[:T])."""

    def __init__(self, device="cpu"):
        import whisper

        self.whisper = whisper
        self.device = device
        self.model = whisper.load_model(os.path.join(CKPT_DIR, "small.pt"), device=device).eval()

    @torch.inference_mode()
    def __call__(self, wav16):
        w = self.whisper
        T = min(1500, int(len(wav16) / 16000 * 1000 / 20) + 1)
        mel = w.log_mel_spectrogram(w.pad_or_trim(torch.from_numpy(np.asarray(wav16, np.float32)))).to(self.device)
        return self.model.embed_audio(mel.unsqueeze(0))[0][:T].cpu().half()  # (T, 768)


# ---------- sequences ----------

def build_sequence(T, text_ids, frames, row, q_ids=None, ctx_ids=None):
    """Input ids (8, L+S-1), audio-feature mask (L+S-1,), targets (8, L+S-1) for one training row.

    row "A1A2": spoken question, audio + text targets; row "A1T2": text targets only, audio inputs are pad_a.
    row "T1A2": typed question (q_ids, Mini-Omni2's text-input layout, T unused), audio + text targets: at
    inference one T1A2 row gives both the text and the audio.
    ctx_ids: retrieved answers (RAG), as a text segment [input_t, ctx..., eot] before the question, audio padded.
    """
    N = len(frames)
    S = N + 8
    text_ids = list(text_ids)
    if len(text_ids) + 1 > S:  # text longer than the audio allows: keep what fits (rare)
        text_ids = text_ids[: S - 1]
    out = np.full((8, S), 0, np.int64)
    for i in range(N_AUDIO):
        col = [PAD_A] * (i + 1) + [int(c) for c in frames[:, i]] + [EOA]
        col += [PAD_A] * (S - len(col))
        out[i] = col[:S]
    out[7] = text_ids + [EOT] + [PAD_T] * (S - len(text_ids) - 1)

    last_a = PAD_A if row == "A1T2" else ANSWER_A
    if row == "T1A2":
        n = len(q_ids)
        prompt = np.zeros((8, n + 3), np.int64)
        for i in range(N_AUDIO):
            prompt[i] = [layershift(PAD_A, i)] * (n + 2) + [layershift(last_a, i)]
        prompt[7] = [INPUT_T] + list(q_ids) + [EOT, ANSWER_T]
    else:
        prompt = np.zeros((8, T + 3), np.int64)
        for i in range(N_AUDIO):
            prompt[i] = [layershift(INPUT_A, i)] + [layershift(PAD_A, i)] * T + [layershift(EOA, i), layershift(last_a, i)]
        prompt[7] = [INPUT_T] + [PAD_T] * T + [EOT, ANSWER_T]

    off = 0
    if ctx_ids is not None and len(ctx_ids):
        pre = np.zeros((8, len(ctx_ids) + 2), np.int64)
        for i in range(N_AUDIO):
            pre[i] = layershift(PAD_A, i)
        pre[7] = [INPUT_T] + list(ctx_ids) + [EOT]
        prompt = np.concatenate([pre, prompt], 1)
        off = pre.shape[1]

    # teacher forcing: the input at answer step s+1 is the output of step s
    fed = np.zeros((8, S - 1), np.int64)
    for i in range(N_AUDIO):
        fed[i] = [layershift(int(t), i) for t in out[i, : S - 1]] if row != "A1T2" else layershift(PAD_A, i)
    fed[7] = out[7, : S - 1]
    ids = np.concatenate([prompt, fed], 1)  # (8, L + S - 1)

    L = prompt.shape[1]
    tgt = np.full((8, L + S - 1), IGNORE, np.int64)
    tgt[:, L - 1 :] = out  # position L-1 (answer_a/answer_t) predicts step 0
    # After EOT, inference forces the text input to pad_t whatever the model predicts, so those text
    # positions are not the model's predictions: no loss there.
    tgt[7, L - 1 + len(text_ids) + 1 :] = IGNORE
    if row == "A1T2":
        tgt[:N_AUDIO] = IGNORE
    mask = np.zeros(L + S - 1, bool)
    if row != "T1A2":
        mask[off + 1 : off + T + 1] = True
    return ids, mask, tgt
