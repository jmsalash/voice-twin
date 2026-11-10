"""Trainable Mini-Omni2 (audio-in/audio-out parts only), with the same parameter names as lit_model.pth.

Full-sequence forward (no KV cache) with PyTorch SDPA attention, tied embedding / LM head like the original.
Load/save use lit_model.pth's key layout, so a fine-tuned checkpoint drops into mini-omni2-web's export/ unchanged.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from omni_lm import (  # noqa: E402
    AUDIO_VOCAB, HEAD, N_AUDIO, N_EMBD, N_GROUPS, N_HEAD, N_LAYER, Q_PER_KV, ROPE_BASE, TEXT_VOCAB, VOCAB,
    GatedMLP, RMSNorm, WhisperAdapter, rope,
)


class TrainAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(N_EMBD, (N_HEAD + 2 * N_GROUPS) * HEAD, bias=True)
        self.proj = nn.Linear(N_HEAD * HEAD, N_EMBD, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = self.attn(x).view(B, T, N_GROUPS, Q_PER_KV + 2, HEAD).permute(0, 2, 3, 1, 4)
        q = rope(qkv[:, :, :Q_PER_KV].reshape(B, N_HEAD, T, HEAD), cos, sin)
        k = rope(qkv[:, :, Q_PER_KV], cos, sin).repeat_interleave(Q_PER_KV, dim=1)
        v = qkv[:, :, Q_PER_KV + 1].repeat_interleave(Q_PER_KV, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(HEAD))
        return self.proj(y.transpose(1, 2).reshape(B, T, N_HEAD * HEAD))


class TrainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_1 = RMSNorm(N_EMBD)
        self.attn = TrainAttention()
        self.norm_2 = RMSNorm(N_EMBD)
        self.mlp = GatedMLP(N_EMBD)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm_1(x), cos, sin)
        return x + self.mlp(self.norm_2(x))


class OmniTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.h = nn.ModuleList(TrainBlock() for _ in range(N_LAYER))
        self.ln_f = RMSNorm(N_EMBD)
        self.whisper_adapter = WhisperAdapter()
        theta = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD, 2).float() / HEAD))
        self.register_buffer("theta", theta, persistent=False)
        self.extra = {}  # weights we don't train but keep for the saved checkpoint (vision adapter)

    def forward(self, ids, feats, mask, checkpointing=False):
        """ids (B, 8, L) shifted ids; feats (B, L, 768) Whisper features aligned to positions; mask (B, L)."""
        B, _, L = ids.shape
        e = self.wte(ids)  # (B, 8, L, 896)
        # The adapter's outputs reach ~21k: times 7 overflows fp16 (max 65504), so the product is taken in e's fp32.
        audio = torch.where(mask.unsqueeze(-1), N_AUDIO * self.whisper_adapter(feats.to(e.dtype)).to(e.dtype), e[:, :N_AUDIO].sum(1))
        x = (audio + e[:, N_AUDIO]) / 8
        pos = torch.arange(L, device=ids.device).float()
        idx = torch.outer(pos, self.theta).repeat(1, 2)
        cos, sin = idx.cos().to(x.dtype), idx.sin().to(x.dtype)
        for blk in self.h:
            if checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        return self.ln_f(x)

    def losses(self, hidden, targets, ignore=-100):
        """Cross-entropy for the text stream and each audio stream, only where targets != ignore."""
        out = {}
        w = self.wte.weight
        tt = targets[:, 7]
        sel = tt != ignore
        if sel.any():
            logits = hidden[sel] @ w[:TEXT_VOCAB].t()
            out["text"] = F.cross_entropy(logits.float(), tt[sel])
        for i in range(N_AUDIO):
            ta = targets[:, i]
            sel = ta != ignore
            if sel.any():
                s = TEXT_VOCAB + i * AUDIO_VOCAB
                logits = hidden[sel] @ w[s : s + AUDIO_VOCAB].t()
                out[f"audio{i}"] = F.cross_entropy(logits.float(), ta[sel])
        return out

    @torch.no_grad()
    def greedy_targets(self, hidden):
        """argmax per stream at every position: (B, 8, L) [audio 0..6 unshifted, text]."""
        w = self.wte.weight
        text = (hidden @ w[:TEXT_VOCAB].t()).argmax(-1)
        audio = [(hidden @ w[TEXT_VOCAB + i * AUDIO_VOCAB : TEXT_VOCAB + (i + 1) * AUDIO_VOCAB].t()).argmax(-1) for i in range(N_AUDIO)]
        return torch.stack(audio + [text], 1)

    # ----- checkpoint I/O in lit_model.pth layout -----

    def load_lit(self, path):
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        mine = {"wte.weight": sd["transformer.wte.weight"], "ln_f.weight": sd["transformer.ln_f.weight"]}
        for k, v in sd.items():
            if k.startswith("transformer.h."):
                mine[k[len("transformer.") :]] = v
            elif k.startswith("whisper_adapter."):
                mine[k] = v
            elif not k.startswith("transformer.") and k != "lm_head.weight":
                self.extra[k] = v  # vision_adapter.* (unused for audio-only, kept so the file stays complete)
        self.load_state_dict({k: v.float() for k, v in mine.items()}, strict=True)
        return self

    def save_lit(self, path):
        sd = {"transformer.wte.weight": self.wte.weight, "transformer.ln_f.weight": self.ln_f.weight}
        for k, v in self.state_dict().items():
            if k.startswith("h."):
                sd["transformer." + k] = v
            elif k.startswith("whisper_adapter."):
                sd[k] = v
        sd["lm_head.weight"] = self.wte.weight  # tied
        sd.update(self.extra)
        torch.save({k: v.detach().float().cpu().contiguous() for k, v in sd.items()}, path)
