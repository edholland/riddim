"""What can this Mac train? Matmul throughput and transformer training
speed in PyTorch (MPS) and MLX, at the sizes a music model would be."""
import sys
import time

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as mopt
import numpy as np
import torch
import torch.nn as tnn

DEV = torch.device("mps")


def sync():
    torch.mps.synchronize()


def bench_torch_matmul(n, dtype, device=DEV, reps=20):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    if device.type == "mps":
        sync()
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    if device.type == "mps":
        sync()
    dt = time.perf_counter() - t
    return 2 * n ** 3 * reps / dt / 1e12


def bench_mlx_matmul(n, dtype, reps=20):
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    for _ in range(3):
        mx.eval(a @ b)
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
        mx.eval(c)
    dt = time.perf_counter() - t
    return 2 * n ** 3 * reps / dt / 1e12


# --------------------------------------------------------------------------
# A GPT-style decoder: the shape of a symbolic music model.
# --------------------------------------------------------------------------

class TBlock(tnn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = tnn.LayerNorm(d), tnn.LayerNorm(d)
        self.attn = tnn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = tnn.Sequential(tnn.Linear(d, 4 * d), tnn.GELU(), tnn.Linear(4 * d, d))

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TGPT(tnn.Module):
    def __init__(self, vocab, d, layers, heads, seq):
        super().__init__()
        self.emb = tnn.Embedding(vocab, d)
        self.pos = tnn.Parameter(torch.zeros(1, seq, d))
        self.blocks = tnn.ModuleList([TBlock(d, heads) for _ in range(layers)])
        self.ln = tnn.LayerNorm(d)
        self.head = tnn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = self.emb(idx) + self.pos[:, :T]
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def bench_torch_train(vocab, d, layers, heads, seq, batch, bf16, steps=6):
    torch.manual_seed(0)
    model = TGPT(vocab, d, layers, heads, seq).to(DEV)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, vocab, (batch, seq), device=DEV)
    y = torch.randint(0, vocab, (batch, seq), device=DEV)

    def step():
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
            logits = model(x)
            loss = tnn.functional.cross_entropy(logits.reshape(-1, vocab).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    dt = time.perf_counter() - t
    mem = torch.mps.driver_allocated_memory() / 1e9
    del model, opt
    torch.mps.empty_cache()
    return params, batch * seq * steps / dt, mem


class MBlock(mnn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = mnn.LayerNorm(d), mnn.LayerNorm(d)
        self.attn = mnn.MultiHeadAttention(d, h)
        self.l1, self.l2 = mnn.Linear(d, 4 * d), mnn.Linear(4 * d, d)

    def __call__(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, mask)
        return x + self.l2(mnn.gelu(self.l1(self.ln2(x))))


class MGPT(mnn.Module):
    def __init__(self, vocab, d, layers, heads, seq):
        super().__init__()
        self.emb = mnn.Embedding(vocab, d)
        self.pos = mnn.Embedding(seq, d)
        self.blocks = [MBlock(d, heads) for _ in range(layers)]
        self.ln = mnn.LayerNorm(d)
        self.head = mnn.Linear(d, vocab, bias=False)

    def __call__(self, idx):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(mx.arange(T))
        mask = mnn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def bench_mlx_train(vocab, d, layers, heads, seq, batch, bf16, steps=6):
    mx.random.seed(0)
    model = MGPT(vocab, d, layers, heads, seq)
    if bf16:
        model.set_dtype(mx.bfloat16)
    params = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    opt = mopt.AdamW(learning_rate=1e-4)
    x = mx.random.randint(0, vocab, (batch, seq))
    y = mx.random.randint(0, vocab, (batch, seq))

    def loss_fn(m):
        return mnn.losses.cross_entropy(m(x).astype(mx.float32), y, reduction="mean")

    grad = mnn.value_and_grad(model, loss_fn)

    def step():
        loss, g = grad(model)
        opt.update(model, g)
        mx.eval(model.parameters(), opt.state, loss)

    for _ in range(2):
        step()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    dt = time.perf_counter() - t
    mem = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
    return params, batch * seq * steps / dt, mem


if __name__ == "__main__":
    print("== matmul throughput (TFLOPS)")
    for n in (4096, 8192):
        cpu = bench_torch_matmul(n, torch.float32, torch.device("cpu"), reps=3) if n == 4096 else float("nan")
        print(f"  n={n}: torch-cpu fp32 {cpu:5.2f} | torch-mps fp32 {bench_torch_matmul(n, torch.float32):5.2f}"
              f" fp16 {bench_torch_matmul(n, torch.float16):5.2f} bf16 {bench_torch_matmul(n, torch.bfloat16):5.2f}"
              f" | mlx fp32 {bench_mlx_matmul(n, mx.float32):5.2f} fp16 {bench_mlx_matmul(n, mx.float16):5.2f}"
              f" bf16 {bench_mlx_matmul(n, mx.bfloat16):5.2f}")
        sys.stdout.flush()

    configs = [
        # name, vocab, d, layers, heads, seq, batch
        ("small  ~25M", 1024, 512, 8, 8, 1024, 16),
        ("medium ~360M", 27000, 1024, 24, 16, 1024, 4),
    ]
    print("\n== transformer training (fwd+bwd+AdamW), tokens/sec")
    for name, vocab, d, L, H, seq, B in configs:
        for bf16 in (False, True):
            tag = "bf16" if bf16 else "fp32"
            p, tps, mem = bench_torch_train(vocab, d, L, H, seq, B, bf16)
            print(f"  {name} torch-mps {tag}: {tps:8.0f} tok/s  (params {p/1e6:.0f}M, batch {B}x{seq}, mem {mem:.1f} GB)")
            sys.stdout.flush()
            p, tps, mem = bench_mlx_train(vocab, d, L, H, seq, B, bf16)
            print(f"  {name} mlx       {tag}: {tps:8.0f} tok/s  (params {p/1e6:.0f}M, peak mem {mem:.1f} GB)")
            sys.stdout.flush()
