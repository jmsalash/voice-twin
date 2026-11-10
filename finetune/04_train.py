"""Step 4: fine-tune Mini-Omni2 on the encoded data (full fine-tune of the language model; Whisper frozen).

Each example is trained as both rows Mini-Omni2 runs at inference: A1A2 (audio + text targets) and
A1T2 (text targets only). CV examples are oversampled relative to the general ones (--cv-weight).
The result is a lit_model.pth-format checkpoint, ready for the server (finetune/package_assistant.py) or the browser (mini-omni2-web: export/build_web_bundle.sh).

Needs a GPU with ~24 GB for the defaults (bf16 autocast, AdamW on fp32 weights, activation checkpointing).
On a 16 GB T4 (free Colab): --amp fp16 --batch 2 --accum 4. --state FILE saves progress every --save-every
steps and resumes from it, for sessions that can disconnect.

  python 04_train.py --out work/ckpt [--epochs 3] [--lr 1e-5] [--batch 8] [--cv-weight 4]
"""

import argparse
import math
import os
import random
import shutil
import time

import numpy as np
import torch

import common as C
from omni_train import OmniTrain


def collate(examples, device):
    """Right-pad a list of (ids, mask, tgt, feat, T) to one batch."""
    Lmax = max(e[0].shape[1] for e in examples)
    B = len(examples)
    ids = np.zeros((B, 8, Lmax), np.int64)
    ids[:, :7] = np.array([C.layershift(C.PAD_A, i) for i in range(7)])[None, :, None]
    ids[:, 7] = C.PAD_T
    mask = np.zeros((B, Lmax), bool)
    tgt = np.full((B, 8, Lmax), C.IGNORE, np.int64)
    feats = torch.zeros(B, Lmax, 768, dtype=torch.float16)
    for b, (i, m, t, f, T) in enumerate(examples):
        L = i.shape[1]
        ids[b, :, :L], mask[b, :L], tgt[b, :, :L] = i, m, t
        pos = np.nonzero(m)[0]  # where the question's audio features go (after any retrieved context)
        feats[b, pos] = f[: len(pos)]
    return (torch.from_numpy(ids).to(device), feats.to(device), torch.from_numpy(mask).to(device), torch.from_numpy(tgt).to(device))


NO_FEAT = torch.zeros(0, 768, dtype=torch.float16)


def expand(rows, rnd, typed=False):
    """Each example -> two training rows (A1A2, A1T2). With typed, also one T1A2 row (the question as text) per
    distinct question, so the model learns what the wordings mean, not only how they sound."""
    out, seen = [], set()
    for r in rows:
        T = r["feat"].shape[0]
        frames = r["frames"].numpy().astype(np.int64)
        ctx = r["ctx"].tolist() if "ctx" in r else None
        for row in ("A1A2", "A1T2"):
            ids, mask, tgt = C.build_sequence(T, r["text"].tolist(), frames, row, ctx_ids=ctx)
            out.append((ids, mask, tgt, r["feat"], T, r["kind"]))
        if typed and "qids" in r and r["question"] not in seen:
            seen.add(r["question"])
            ids, mask, tgt = C.build_sequence(0, r["text"].tolist(), frames, "T1A2", q_ids=r["qids"].tolist(), ctx_ids=ctx)
            out.append((ids, mask, tgt, NO_FEAT, 0, r["kind"]))
    rnd.shuffle(out)
    return out


@torch.no_grad()
def evaluate(model, rows, device, batch, dtype=None):
    model.eval()
    tot = {}
    n = 0
    for k in range(0, len(rows), batch):
        b = collate([r[:5] for r in rows[k : k + batch]], device)
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            h = model(b[0], b[1], b[2])
            losses = model.losses(h, b[3])
        for key, v in losses.items():
            tot[key] = tot.get(key, 0.0) + v.item()
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(C.WORK, "encoded"))
    ap.add_argument("--init", default=os.path.join(C.CKPT_DIR, "lit_model.pth"))
    ap.add_argument("--out", default=os.path.join(C.WORK, "ckpt"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--cv-weight", type=int, default=4, help="how many times each CV example appears per epoch")
    ap.add_argument("--text-weight", type=float, default=1.0)
    ap.add_argument("--audio-weight", type=float, default=1.0)
    ap.add_argument("--typed", action="store_true", help="also train each distinct question typed (T1A2 rows)")
    ap.add_argument("--train-adapter", action="store_true", help="also train the Whisper adapter (default: frozen)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"],
                    help="fp16 for GPUs without bf16 such as the T4 (with loss scaling); none for Pascal")
    ap.add_argument("--state", default="", help="training state file: saved every --save-every steps, resumed if present")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"], help="sgd only for smoke tests on small GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0, help="share of distinct training questions held out for early "
                    "stopping (all their voices); the eval set stays untouched for the final test")
    ap.add_argument("--val-every", type=int, default=200, help="steps between validation checks")
    ap.add_argument("--patience", type=int, default=4, help="checks without improvement before stopping")
    ap.add_argument("--save-last", action="store_true", help="with --val-frac: also keep the final weights, in OUT/last")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp) if args.device == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    rnd = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train_rows = torch.load(os.path.join(args.data, "train.pt"), weights_only=False)
    eval_rows = torch.load(os.path.join(args.data, "eval.pt"), weights_only=False) if os.path.exists(os.path.join(args.data, "eval.pt")) else []
    val_seq = []
    if args.val_frac > 0:  # by question wording, so no voice of a validation question is trained on
        qs = sorted({r["question"] for r in train_rows})
        val_q = set(random.Random(args.seed + 7).sample(qs, max(1, int(len(qs) * args.val_frac))))
        val_rows = [r for r in train_rows if r["question"] in val_q]
        train_rows = [r for r in train_rows if r["question"] not in val_q]
        val_seq = expand(val_rows, random.Random(args.seed + 8), args.typed)
        print(f"validation: {len(val_q)} questions ({len(val_rows)} examples) held out for early stopping")
    train_rows = [r for r in train_rows for _ in range(args.cv_weight if r["kind"] == "cv" else 1)]
    eval_seq = expand(eval_rows, rnd, args.typed)
    print(f"train examples (with CV oversampling): {len(train_rows)}, eval examples: {len(eval_rows)}")

    model = OmniTrain().load_lit(args.init).to(args.device)
    model.whisper_adapter.requires_grad_(args.train_adapter)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)) if args.optim == "adamw"
           else torch.optim.SGD(params, lr=args.lr))
    n_typed = len({r["question"] for r in train_rows if "qids" in r}) if args.typed else 0
    steps_per_epoch = math.ceil((2 * len(train_rows) + n_typed) / (args.batch * args.accum))
    total = int(steps_per_epoch * args.epochs) if not args.max_steps else args.max_steps
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    os.makedirs(args.out, exist_ok=True)
    print(f"{total} optimizer steps, {steps_per_epoch} per epoch")
    step = 0
    best, bad = float("inf"), 0  # early stopping: best validation score so far, checks since it
    if args.state and os.path.exists(args.state):
        st = torch.load(args.state, map_location="cpu", weights_only=False)  # not next to the model on the GPU
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        best, bad = st.get("best", best), st.get("bad", 0)
        rnd.seed(args.seed + step)
        print(f"resumed at step {step}", flush=True)

    def save_state():
        tmp = args.state + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "bad": bad}, tmp)
        os.replace(tmp, args.state)  # never leave a half-written state behind

    if eval_seq and step == 0:
        print("eval before training:", {k: round(v, 3) for k, v in evaluate(model, eval_seq, args.device, args.batch, dtype).items()})

    def save_model(out=args.out):
        os.makedirs(out, exist_ok=True)
        model.save_lit(os.path.join(out, "lit_model.pth"))
        for f in ("model_config.yaml", "tokenizer.json", "tokenizer_config.json", "small.pt"):
            src = os.path.join(C.CKPT_DIR, f)
            dst = os.path.join(out, f)
            if os.path.exists(src) and not os.path.exists(dst):
                # small files are copied, so the folder still works after being downloaded; the 460 MB Whisper is linked
                os.symlink(src, dst) if f == "small.pt" else shutil.copy(src, dst)

    t0 = time.time()
    stopped = False
    model.train()
    while step < total and not stopped:
        seqs = expand(train_rows, rnd, args.typed)
        # bucket by length to reduce padding
        seqs.sort(key=lambda s: s[0].shape[1])
        batches = [seqs[k : k + args.batch] for k in range(0, len(seqs), args.batch)]
        rnd.shuffle(batches)
        for bi, batch in enumerate(batches):
            b = collate([s[:5] for s in batch], args.device)
            with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                h = model(b[0], b[1], b[2], checkpointing=True)
                losses = model.losses(h, b[3])
            loss = sum((args.text_weight if k == "text" else args.audio_weight) * v for k, v in losses.items()) / args.accum
            scaler.scale(loss).backward()
            if (bi + 1) % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} loss {loss.item() * args.accum:.3f} text {losses.get('text', torch.tensor(0)).item():.3f} "
                      f"audio0 {losses.get('audio0', torch.tensor(0)).item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if step % steps_per_epoch == 0 and eval_seq:
                print("eval:", {k: round(v, 3) for k, v in evaluate(model, eval_seq, args.device, args.batch, dtype).items()}, flush=True)
            if val_seq and step % args.val_every == 0:
                v = evaluate(model, val_seq, args.device, args.batch, dtype)
                score = v.get("text", 0) + sum(x for k, x in v.items() if k.startswith("audio")) / 7
                if score < best:
                    best, bad = score, 0
                    save_model()
                    print(f"validation {score:.4f} (text {v.get('text', 0):.3f}): best so far, saved", flush=True)
                else:
                    bad += 1
                    print(f"validation {score:.4f}: no better than {best:.4f} ({bad}/{args.patience})", flush=True)
                    if bad >= args.patience:
                        print("early stop", flush=True)
                        stopped = True
                        break
            if args.state and step % args.save_every == 0 and step < total:
                save_state()
            if step >= total:
                break
    path = os.path.join(args.out, "lit_model.pth")
    if val_seq:  # the best checkpoint is already saved; measure that one
        v = evaluate(model, val_seq, args.device, args.batch, dtype)
        score = v.get("text", 0) + sum(x for k, x in v.items() if k.startswith("audio")) / 7
        if score < best:
            best = score
            save_model()
        elif args.save_last:  # the best is earlier: keep these final weights too, to compare
            save_model(os.path.join(args.out, "last"))
            print(f"kept the last weights too (validation {score:.4f}) in", os.path.join(args.out, "last"))
        del model
        torch.cuda.empty_cache()
        model = OmniTrain().load_lit(path).to(args.device)
        print(f"kept the best checkpoint (validation {best:.4f})")
    else:
        save_model()
    if eval_seq:
        print("eval after training:", {k: round(v, 3) for k, v in evaluate(model, eval_seq, args.device, args.batch, dtype).items()})
    print("saved", path)


if __name__ == "__main__":
    main()
