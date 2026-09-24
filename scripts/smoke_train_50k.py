#!/usr/bin/env python3
"""Pythia M50M training smoke test (200 steps) on FULL 50k-tokenized OWT.

Streams from data/openwebtext_combined_bpe_owt50k.bin via memmap — no RAM
load, no GPU load, just random-batch sampling from disk.
"""
import os, sys, json, time, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Config — gpt-neox / Pythia-style: 6L × 512H × 8 heads
# EleutherAI pythia-70m shape (6/512/8/4) ~52M @ vocab 50k
# ---------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.vocab_size = 50256
        self.context_length = 512
        self.embedding_dim = 512
        self.num_layers = 6
        self.num_heads = 8
        self.ff_ratio = 4
        self.dropout = 0.1
        self.rope_base = 10000.0

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads

    def num_params(self):
        d = self.embedding_dim
        ff = d * self.ff_ratio
        per_layer = 4 * d * d + 2 * d * ff
        return (
            self.vocab_size * d
            + self.context_length * d
            + self.num_layers * per_layer
            + 2 * d
            + d * self.vocab_size
            + self.vocab_size
        )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def make_rope_cache(cfg):
    inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(cfg.embedding_dim // 2).float() / (cfg.embedding_dim // 2)))
    pos = torch.arange(cfg.context_length, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', pos, inv_freq)
    cos = freqs.cos().half()
    sin = freqs.sin().half()
    return cos, sin


def apply_rope(q, k, cos, sin):
    B, T, D = q.shape
    half_d = D // 2
    q1, q2 = q[..., :half_d].float(), q[..., half_d:].float()
    k1, k2 = k[..., :half_d].float(), k[..., half_d:].float()
    c = cos[:T].view(1, T, half_d).expand(B, T, half_d)
    s = sin[:T].view(1, T, half_d).expand(B, T, half_d)
    qr1 = q1 * c - q2 * s
    qr2 = q2 * c + q1 * s
    kr1 = k1 * c - k2 * s
    kr2 = k2 * c + k1 * s
    return torch.cat([qr1.half(), qr2.half()], dim=-1), torch.cat([kr1.half(), kr2.half()], dim=-1)


# ---------------------------------------------------------------------------
# GPT-NeoX layer
# ---------------------------------------------------------------------------
class GPTNeXtLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.embedding_dim, bias=False)
        self.wq = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wk = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wv = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wo = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(cfg.embedding_dim, bias=False)
        self.gate_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim * cfg.ff_ratio, bias=False)
        self.up_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim * cfg.ff_ratio, bias=False)
        self.down_proj = nn.Linear(cfg.embedding_dim * cfg.ff_ratio, cfg.embedding_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self._d = cfg.embedding_dim
        self._ff = cfg.embedding_dim * cfg.ff_ratio
        self._nh = cfg.num_heads
        self._head_d = self._d // self._nh

    def forward(self, x, cos, sin):
        r = self.attention_norm(x)
        q, k, v = self.wq(r), self.wk(r), self.wv(r)
        q, k = apply_rope(q, k, cos, sin)
        B, T, D = q.shape
        n_heads, head_d = self._nh, self._head_d
        q = q.view(B, T, n_heads, head_d).transpose(1, 2)
        k = k.view(B, T, n_heads, head_d).transpose(1, 2)
        v = v.view(B, T, n_heads, head_d).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout.p if hasattr(self.dropout, 'p') else 0)
        att = att.transpose(1, 2).contiguous().view(B, T, D)
        att = self.dropout(self.wo(att))
        out = x + att
        r = self.ffn_norm(out)
        gate = self.gate_proj(r)
        up = self.up_proj(r)
        ffn_out = self.dropout(self.down_proj(F.silu(gate) * up))
        return out + ffn_out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GPTNeXtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_embeddings = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.layers = nn.ModuleList([GPTNeXtLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.embedding_dim, bias=False)
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        cos_full, sin_full = make_rope_cache(cfg)
        self.register_buffer('_rope_cos_full', cos_full, persistent=False)
        self.register_buffer('_rope_sin_full', sin_full, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.tok_embeddings(idx)
        pos_emb = self.pos_embeddings(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        cos = self._rope_cos_full[:T].to(idx.device)
        sin = self._rope_sin_full[:T].to(idx.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Data loading — memmap, sample batches without loading to GPU
# ---------------------------------------------------------------------------
def load_tokens_memmap(path):
    n = os.path.getsize(path) // 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.memmap(path, dtype="<i4", mode="r", shape=(n,))
    print(f"  Memmap: {os.path.basename(path)}: {n:,} tokens ({n*4/1e9:.1f}GB on disk, 0 RAM)")
    return arr


def get_batch(memmap_arr, batch_size, context_length, device):
    """Sample random batches from the memmap without loading to GPU."""
    total = len(memmap_arr) - context_length - 1
    ix = np.random.randint(0, total, size=batch_size, dtype=np.int64)
    # Gather slices — small CPU tensor, then move to GPU
    offsets = np.expand_dims(ix, 1) + np.arange(context_length, dtype=np.int64)
    x_np = memmap_arr[offsets]  # (B, T) on CPU
    y_np = memmap_arr[offsets + 1]
    x = torch.from_numpy(x_np).to(device).long()
    y = torch.from_numpy(y_np).to(device).long()
    return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA: {name}, VRAM: {vram:.1f}GB\n")
    else:
        print("WARNING: Running on CPU — will be very slow.\n")
        return 1

    cfg = Config()
    print(f"Model: Pythia GPT-NeXt style {cfg.num_layers}L × {cfg.embedding_dim}H × {cfg.num_heads}H")
    print(f"Vocab: {cfg.vocab_size}, Context: {cfg.context_length}")
    print(f"Params: {cfg.num_params()/1e6:.1f}M\n")

    # Full 50k OWT — memmap, never loaded into RAM or GPU
    bin_path = 'D:/Projects/llm-lab/data/openwebtext_combined_bpe_owt50k.bin'
    if not os.path.exists(bin_path):
        print(f"ERROR: {bin_path} not found!")
        return 1

    tok_arr = load_tokens_memmap(bin_path)
    train_split = int(len(tok_arr) * 0.95)
    print(f"Train: 95% ({train_split/1e6:.0f}M tokens) | Val: 5% ({(len(tok_arr)-train_split)/1e6:.0f}M tokens)\n")

    model = GPTNeXtModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, eps=1e-8)

    STEPS = 20000
    BATCH = 32
    VAL_EVERY = 200
    CKPT_EVERY = 500
    PATIENCE_FRAC = 0.15
    MIN_STEPS_FRAC = 0.6
    DEGRADE_FRAC = 0.15
    ctx = cfg.context_length

    run_dir = 'D:/Projects/llm-lab/runs/smoke_50k_pythia_m50m'
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'loss.jsonl')
    log_f = open(log_path, 'w')
    header = {
        "args": {"steps": STEPS, "batch": BATCH, "context_length": ctx, "lr": 3e-4, "dropout": cfg.dropout,
                 "early_stop": {"patience_frac": PATIENCE_FRAC, "min_steps_frac": MIN_STEPS_FRAC, "degrade_frac": DEGRADE_FRAC}},
        "cfg": {"vocab_size": cfg.vocab_size, "embedding_dim": cfg.embedding_dim,
                "num_layers": cfg.num_layers, "num_heads": cfg.num_heads,
                "params_m": round(cfg.num_params()/1e6, 2)},
        "device": str(device),
        "corpus": "openwebtext_combined_bpe_owt50k.bin (FULL 50k OWT, 8.55B tokens)",
    }
    log_f.write(json.dumps(header) + "\n")
    log_f.flush()

    print("=" * 60)
    print(f"SMOKE TEST — {STEPS} steps, batch={BATCH}, ctx={ctx}")
    print(f"Corpus: FULL 50k OWT (8.55B tokens)")
    print(f"LR: {3e-4:.0e}, Val every: {VAL_EVERY}, Ckpt every: {CKPT_EVERY}")
    print(f"Early stop: patience={int(STEPS*PATIENCE_FRAC)} steps, min={int(STEPS*MIN_STEPS_FRAC)} steps, degrade={DEGRADE_FRAC}")
    print("=" * 60)

    # Early stop state
    best_val = float('inf')
    best_step = 0
    bad_checks = 0
    min_steps = int(STEPS * MIN_STEPS_FRAC)
    patience_steps = int(STEPS * PATIENCE_FRAC)
    degrade_limit = 1.0 + DEGRADE_FRAC
    early_stopped = False

    for step in range(1, STEPS + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(tok_arr, BATCH, ctx, device)

        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0
        toks_per_sec = (BATCH * ctx) / dt

        if step % 10 == 0 or step == 1 or step == STEPS:
            print(f"Step {step:4d}/{STEPS} | loss={loss.item():.4f} | {toks_per_sec:,.0f} toks/s | {dt:.2f}s")

        log_entry = {"step": step, "train_loss": round(loss.item(), 6),
                     "lr": optimizer.param_groups[0]['lr'], "toks_per_sec": round(toks_per_sec, 0)}
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        if step % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    vx, vy = get_batch(tok_arr, BATCH, ctx, device)
                    with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                        _, vloss = model(vx, vy)
                    val_losses.append(vloss.item())
            val_mean = sum(val_losses) / len(val_losses)
            log_entry["val_loss"] = round(val_mean, 6)

            # Early stop logic
            if val_mean < best_val:
                best_val = val_mean
                best_step = step
                bad_checks = 0
                best_path = os.path.join(run_dir, 'best.pt')
                torch.save({'step': step, 'model': {k: v.cpu() for k, v in model.state_dict().items()},
                            'val_loss': val_mean}, best_path)
                print(f"         | val_loss={val_mean:.4f} (best @ step {step}) ✓")
            else:
                bad_checks += 1
                print(f"         | val_loss={val_mean:.4f} (best {best_val:.4f} @ step {best_step}, {bad_checks} bad)")

            if val_mean > best_val * degrade_limit:
                print(f"\n!! DEGRADED: val {val_mean:.4f} > {degrade_limit:.2f}x best {best_val:.4f}")
                log_entry["early_stop"] = "degraded"
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.close()
                model.train()
                break

            if step >= min_steps and bad_checks * VAL_EVERY >= patience_steps:
                print(f"\n!! EARLY STOP: no improvement for {patience_steps} steps (best @ {best_step})")
                log_entry["early_stop"] = "patience"
                log_f.write(json.dumps(log_entry) + "\n")
                log_f.close()
                model.train()
                early_stopped = True
                break

            model.train()

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(run_dir, f'step{step}.pt')
            torch.save({'step': step, 'model': {k: v.cpu() for k, v in model.state_dict().items()},
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if device.type == "cuda" else {}}, ckpt_path)
            size_mb = os.path.getsize(ckpt_path) / 1e6
            print(f"         | checkpoint saved: step{step}.pt ({size_mb:.0f}MB)\n")

    log_f.close()

    print("\n" + "=" * 60)
    if early_stopped:
        print("EARLY STOPPED")
    else:
        print("RUN COMPLETE")
    print(f"Best val loss: {best_val:.4f} @ step {best_step}")
    print("=" * 60)
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())