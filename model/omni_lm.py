"""Mini-Omni2's language model, rewritten for ONNX export (audio-in, audio-out only).

Same weights and math as the litgpt model in reference/litgpt/model.py, with three changes
that make it exportable and cheap to run in the browser:

- The KV cache is explicit: past keys/values go in, present keys/values come out.
  Keys/values are stored per query group (2 groups) and expanded to 14 heads inside.
- The token embedding lookup and the 8-stream averaging are done by the caller
  (in the browser, from a separate table), so the graph takes `inputs_embeds`.
  This keeps one copy of the 181120 x 896 table out of the graph.
- Only the last position goes through the LM head, and greedy decoding (Mini-Omni2 uses
  top_k=1) happens in the graph: it returns 8 token ids per row instead of 181k logits.

The vision adapter is dropped. The Whisper adapter is a separate module (`WhisperAdapter`).
"""

import math

import torch
import torch.nn as nn

N_EMBD = 896
N_LAYER = 24
N_HEAD = 14
N_GROUPS = 2
Q_PER_KV = N_HEAD // N_GROUPS
HEAD = 64
INTER = 4864
EPS = 1e-6
ROPE_BASE = 1_000_000

TEXT_VOCAB = 152000  # text logits slice (151936 tokens + 64 special)
AUDIO_VOCAB = 4160  # per SNAC layer (4096 codes + 64 special)
N_AUDIO = 7
VOCAB = TEXT_VOCAB + N_AUDIO * AUDIO_VOCAB  # 181120
WHISPER_DIM = 768

# Special ids (unshifted). Audio ids for layer i are shifted by TEXT_VOCAB + i * AUDIO_VOCAB.
EOT, PAD_T, INPUT_T, ANSWER_T = 151936, 151937, 151938, 151939
EOA, PAD_A, INPUT_A, ANSWER_A = 4096, 4097, 4098, 4099


def layershift(token: int, layer: int) -> int:
    return token + TEXT_VOCAB + layer * AUDIO_VOCAB


class RMSNorm(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
        return x32.to(x.dtype) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(n_in, INTER, bias=False)
        self.fc_2 = nn.Linear(n_in, INTER, bias=False)
        self.proj = nn.Linear(INTER, N_EMBD, bias=False)

    def forward(self, x):
        return self.proj(nn.functional.silu(self.fc_1(x)) * self.fc_2(x))


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(N_EMBD, (N_HEAD + 2 * N_GROUPS) * HEAD, bias=True)
        self.proj = nn.Linear(N_HEAD * HEAD, N_EMBD, bias=False)

    def forward(self, x, cos, sin, allowed, past_k, past_v):
        B, T, _ = x.shape
        # litgpt layout: per query group, [7 queries, 1 key, 1 value]
        qkv = self.attn(x).view(B, T, N_GROUPS, Q_PER_KV + 2, HEAD).permute(0, 2, 3, 1, 4)
        q = qkv[:, :, :Q_PER_KV].reshape(B, N_HEAD, T, HEAD)
        k = qkv[:, :, Q_PER_KV]  # (B, G, T, hs)
        v = qkv[:, :, Q_PER_KV + 1]
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        k = torch.cat((past_k, k), dim=2)
        v = torch.cat((past_v, v), dim=2)
        S = k.shape[2]
        kk = k.unsqueeze(2).expand(B, N_GROUPS, Q_PER_KV, S, HEAD).reshape(B, N_HEAD, S, HEAD)
        vv = v.unsqueeze(2).expand(B, N_GROUPS, Q_PER_KV, S, HEAD).reshape(B, N_HEAD, S, HEAD)
        scores = (q @ kk.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD))
        scores = scores.masked_fill(~allowed, float("-inf"))
        y = torch.softmax(scores, dim=-1) @ vv
        y = y.transpose(1, 2).reshape(B, T, N_HEAD * HEAD)
        return self.proj(y), k, v


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(N_EMBD)
        self.attn = Attention()
        self.norm_2 = RMSNorm(N_EMBD)
        self.mlp = GatedMLP(N_EMBD)

    def forward(self, x, cos, sin, allowed, past_k, past_v):
        a, k, v = self.attn(self.norm_1(x), cos, sin, allowed, past_k, past_v)
        x = x + a
        x = x + self.mlp(self.norm_2(x))
        return x, k, v


class OmniDecoder(nn.Module):
    """inputs_embeds (B, T, 896) + past KV -> next tokens (B, 8) + present KV.

    Next tokens are [audio layer 0..6 (unshifted, 0..4159), text (0..151999)].
    """

    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = RMSNorm(N_EMBD)
        # Explicit (untied) copy of the LM head, stored as (896, VOCAB) so it is a plain MatMul initializer
        # that weight-only quantizers recognise.
        self.lm_head_t = nn.Parameter(torch.zeros(N_EMBD, VOCAB))
        theta = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD, 2).float() / HEAD))
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, inputs_embeds, *past):
        B, T, _ = inputs_embeds.shape
        P = past[0].shape[2]
        pos = torch.arange(T, device=inputs_embeds.device).float() + P
        idx_theta = torch.outer(pos, self.theta).repeat(1, 2)  # (T, hs)
        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
        q_pos = torch.arange(T, device=inputs_embeds.device).unsqueeze(1) + P
        k_pos = torch.arange(P + T, device=inputs_embeds.device).unsqueeze(0)
        allowed = k_pos <= q_pos  # (T, S)

        x = inputs_embeds
        present = []
        for i, block in enumerate(self.h):
            x, k, v = block(x, cos, sin, allowed, past[2 * i], past[2 * i + 1])
            present += [k, v]

        h = self.ln_f(x[:, -1:, :])
        logits = (h @ self.lm_head_t)[:, 0]  # (B, VOCAB)
        text = logits[:, :TEXT_VOCAB].argmax(-1, keepdim=True)
        audio = logits[:, TEXT_VOCAB:].reshape(B, N_AUDIO, AUDIO_VOCAB).argmax(-1)
        return (torch.cat((audio, text), dim=1), *present)


class WhisperAdapter(nn.Module):
    """Whisper-small encoder features (B, T, 768) -> LLM embedding space (B, T, 896)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(WHISPER_DIM, INTER, bias=False)
        self.fc_2 = nn.Linear(WHISPER_DIM, INTER, bias=False)
        self.proj = nn.Linear(INTER, N_EMBD, bias=False)

    def forward(self, x):
        return self.proj(nn.functional.silu(self.fc_1(x)) * self.fc_2(x))


def load(lit_model_path: str):
    """Returns (decoder, adapter, wte) with weights from Mini-Omni2's lit_model.pth."""
    sd = torch.load(lit_model_path, map_location="cpu", mmap=True, weights_only=True)
    wte = sd["transformer.wte.weight"].float()  # (VOCAB, 896), tied with lm_head

    dec = OmniDecoder()
    dec_sd = {"ln_f.weight": sd["transformer.ln_f.weight"], "lm_head_t": wte.t().contiguous()}
    for k, v in sd.items():
        if k.startswith("transformer.h."):
            dec_sd[k[len("transformer.") :]] = v
    dec.load_state_dict(dec_sd, strict=True)

    ad = WhisperAdapter()
    ad.load_state_dict({k[len("whisper_adapter.") :]: v for k, v in sd.items() if k.startswith("whisper_adapter.")}, strict=True)

    unused = [k for k in sd if not (k.startswith("transformer.") or k.startswith("whisper_adapter.") or k.startswith("vision_adapter.") or k == "lm_head.weight")]
    assert not unused, f"unexpected weights: {unused}"
    return dec.float().eval(), ad.float().eval(), wte


def embed(wte, ids, audio_embeds=None, audio_mask=None):
    """The 8-stream input embedding, as the browser computes it.

    ids: (B, 8, T) shifted ids. audio_embeds: (B, T, 896) adapter output aligned to positions,
    audio_mask: (B, T) bool; where set, the 7 audio streams use audio_embeds instead of their token.
    """
    e = wte[ids]  # (B, 8, T, 896)
    audio_sum = e[:, :N_AUDIO].sum(1)
    if audio_embeds is not None:
        audio_sum = torch.where(audio_mask.unsqueeze(-1), N_AUDIO * audio_embeds, audio_sum)
    return (audio_sum + e[:, N_AUDIO]) / 8


def empty_past(batch: int, dtype=torch.float32):
    return [torch.zeros(batch, N_GROUPS, 0, HEAD, dtype=dtype) for _ in range(2 * N_LAYER)]
