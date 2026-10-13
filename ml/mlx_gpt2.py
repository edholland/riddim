"""GPT-2 in MLX, weight-for-weight with Hugging Face's GPT2LMHeadModel, so a
checkpoint trains here and samples in PyTorch (riddim's worker) unchanged.

Weights stay in Hugging Face's layout (Conv1D: [in, out], used as x @ W),
so loading and saving are a straight copy of the safetensors file. The
Anticipatory Music Transformer uses `scale_attn_by_inverse_layer_idx`:
layer i divides its attention scores by i + 1 as well as sqrt(head dim).

Training keeps fp32 master weights and computes in bf16 (`compute_dtype`):
a fine-tuning step at lr 1e-5 is below bf16's resolution for most weights,
so pure-bf16 weights would simply not move.
"""
import json
import math
import os
import shutil

import mlx.core as mx
import mlx.nn as nn


def _ln(x, w, b, eps):
    return mx.fast.layer_norm(x, w, b, eps)


def _gelu_new(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class GPT2(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d, L = config["n_embd"], config["n_layer"]
        self.n_head = config["n_head"]
        self.eps = config.get("layer_norm_epsilon", 1e-5)
        self.inv_layer = config.get("scale_attn_by_inverse_layer_idx", False)
        self.dropout = 0.0
        self.compute_dtype = mx.float32
        z = mx.zeros
        self.wte = z((config["vocab_size"], d))
        self.wpe = z((config["n_positions"], d))
        self.h = [{
            "ln_1": {"weight": z((d,)), "bias": z((d,))},
            "attn": {"c_attn": {"weight": z((d, 3 * d)), "bias": z((3 * d,))},
                     "c_proj": {"weight": z((d, d)), "bias": z((d,))}},
            "ln_2": {"weight": z((d,)), "bias": z((d,))},
            "mlp": {"c_fc": {"weight": z((d, 4 * d)), "bias": z((4 * d,))},
                    "c_proj": {"weight": z((4 * d, d)), "bias": z((d,))}},
        } for _ in range(L)]
        self.ln_f = {"weight": z((d,)), "bias": z((d,))}

    def _drop(self, x):
        if self.dropout and self.training:
            keep = mx.random.bernoulli(1 - self.dropout, x.shape)
            return x * keep.astype(x.dtype) / (1 - self.dropout)
        return x

    def __call__(self, idx):
        B, T = idx.shape
        cd = self.compute_dtype
        c = (lambda a: a.astype(cd)) if cd != mx.float32 else (lambda a: a)
        hd = self.config["n_embd"] // self.n_head
        x = c(self.wte)[idx] + c(self.wpe)[:T]
        x = self._drop(x)
        for i, blk in enumerate(self.h):
            a = blk["attn"]
            y = _ln(x, c(blk["ln_1"]["weight"]), c(blk["ln_1"]["bias"]), self.eps)
            qkv = y @ c(a["c_attn"]["weight"]) + c(a["c_attn"]["bias"])
            q, k, v = mx.split(qkv, 3, axis=-1)
            q, k, v = (t.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
            scale = 1.0 / math.sqrt(hd) / ((i + 1) if self.inv_layer else 1)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
            o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
            x = x + self._drop(o @ c(a["c_proj"]["weight"]) + c(a["c_proj"]["bias"]))
            m = blk["mlp"]
            y = _ln(x, c(blk["ln_2"]["weight"]), c(blk["ln_2"]["bias"]), self.eps)
            y = _gelu_new(y @ c(m["c_fc"]["weight"]) + c(m["c_fc"]["bias"]))
            x = x + self._drop(y @ c(m["c_proj"]["weight"]) + c(m["c_proj"]["bias"]))
        x = _ln(x, c(self.ln_f["weight"]), c(self.ln_f["bias"]), self.eps)
        return x @ c(self.wte).T                     # tied head


def _hf_to_tree(weights: dict, n_layer: int) -> dict:
    t = {"wte": weights["transformer.wte.weight"], "wpe": weights["transformer.wpe.weight"],
         "ln_f": {"weight": weights["transformer.ln_f.weight"], "bias": weights["transformer.ln_f.bias"]},
         "h": []}
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        g = lambda k: weights[p + k]  # noqa: E731
        t["h"].append({
            "ln_1": {"weight": g("ln_1.weight"), "bias": g("ln_1.bias")},
            "attn": {"c_attn": {"weight": g("attn.c_attn.weight"), "bias": g("attn.c_attn.bias")},
                     "c_proj": {"weight": g("attn.c_proj.weight"), "bias": g("attn.c_proj.bias")}},
            "ln_2": {"weight": g("ln_2.weight"), "bias": g("ln_2.bias")},
            "mlp": {"c_fc": {"weight": g("mlp.c_fc.weight"), "bias": g("mlp.c_fc.bias")},
                    "c_proj": {"weight": g("mlp.c_proj.weight"), "bias": g("mlp.c_proj.bias")}},
        })
    return t


def load(model_dir: str) -> GPT2:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    weights = mx.load(os.path.join(model_dir, "model.safetensors"))
    model = GPT2(config)
    model.update(_hf_to_tree({k: v.astype(mx.float32) for k, v in weights.items()}, config["n_layer"]))
    mx.eval(model.parameters())
    model.source_dir = model_dir
    return model


def save(model: GPT2, out_dir: str) -> None:
    """A Hugging Face checkpoint: config, generation config, safetensors."""
    os.makedirs(out_dir, exist_ok=True)
    p = model.parameters()
    w = {"transformer.wte.weight": p["wte"], "transformer.wpe.weight": p["wpe"],
         "transformer.ln_f.weight": p["ln_f"]["weight"], "transformer.ln_f.bias": p["ln_f"]["bias"]}
    for i, blk in enumerate(p["h"]):
        pre = f"transformer.h.{i}."
        for a, b in (("ln_1", None), ("ln_2", None)):
            w[pre + a + ".weight"] = blk[a]["weight"]
            w[pre + a + ".bias"] = blk[a]["bias"]
        for grp, sub in (("attn", "c_attn"), ("attn", "c_proj"), ("mlp", "c_fc"), ("mlp", "c_proj")):
            w[f"{pre}{grp}.{sub}.weight"] = blk[grp][sub]["weight"]
            w[f"{pre}{grp}.{sub}.bias"] = blk[grp][sub]["bias"]
    tmp = os.path.join(out_dir, "model.safetensors.tmp.safetensors")
    mx.save_safetensors(tmp, w, metadata={"format": "pt"})
    os.replace(tmp, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(model.config, f, indent=2)
    src = getattr(model, "source_dir", None)
    if src and os.path.exists(os.path.join(src, "generation_config.json")):
        shutil.copy(os.path.join(src, "generation_config.json"), out_dir)
