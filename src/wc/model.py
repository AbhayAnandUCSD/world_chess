"""Minimal nanoGPT re-implementation matching Karvonen's chess checkpoints.

One implementation is used for training, scoring AND probing, so next-char
probabilities and activations come from identical code (no TransformerLens
conversion step whose LayerNorm folding could drift).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
CODE_ROOT = Path(__file__).resolve().parents[2]
# WC_ROOT selects an experiment folder (configs/, data/, ckpts/, results/, runs/); code is shared.
ROOT = Path(os.environ.get("WC_ROOT", CODE_ROOT)).resolve()
CKPT_DIR = ROOT / "ckpts"
HF_REPO = "adamkarvonen/chess_llms"

# Exact vocab from chess_llm_interpretability/models/meta.pkl (32 chars, sorted).
ITOS = [" ", "#", "+", "-", ".", *list("0123456789"), ";", "=", "B", "K", "N", "O", "Q", "R", *list("abcdefgh"), "x"]
STOI = {c: i for i, c in enumerate(ITOS)}
assert len(ITOS) == 32 and STOI[";"] == 15 and STOI["x"] == 31


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPTConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1023
    vocab_size: int = 32
    bias: bool = False
    dropout: float = 0.0


class LayerNorm(nn.Module):
    def __init__(self, n, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Attn(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.n_head = c.n_head
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop.p if self.training else 0.0)
        return self.drop(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(c.n_embd, c.bias)
        self.attn = Attn(c)
        self.ln_2 = LayerNorm(c.n_embd, c.bias)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=LayerNorm(c.n_embd, c.bias),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.emb_drop = nn.Dropout(c.dropout)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as nanoGPT

    def forward(self, idx, return_resid: bool = False, edit=None):
        """idx: (B,T) long. Returns logits (B,T,V) and optionally the residual
        stream list [embed, after block0, ..., after block7] (each (B,T,C))."""
        T = idx.shape[1]
        assert T <= self.config.block_size, f"sequence {T} > context {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        if self.training and self.emb_drop.p > 0:
            x = self.emb_drop(x)
        resid = [x] if return_resid else None
        # edit: (resid_index, positions LongTensor (B,), vectors (B,C)) -> add vec to resid at that pos
        for li, blk in enumerate(self.transformer.h):
            x = blk(x)
            if edit is not None and li + 1 == edit[0]:
                x = x.clone(); x[torch.arange(x.shape[0], device=x.device), edit[1]] += edit[2]
            if return_resid:
                resid.append(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        return (logits, resid) if return_resid else logits


def _ckpt_path(name: str) -> Path:
    p = CKPT_DIR / name
    if not p.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, name, local_dir=str(CKPT_DIR))
    return p


def load_model(name_or_path: str = "lichess_8layers_ckpt_no_optimizer.pt", device: str | None = None) -> GPT:
    """Load a Karvonen HF checkpoint by name, or a local fine-tuned .pt we saved."""
    device = device or pick_device()
    p = Path(name_or_path)
    if not p.exists():
        p = _ckpt_path(name_or_path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    args = {k: v for k, v in ck["model_args"].items() if k in GPTConfig.__dataclass_fields__}
    m = GPT(GPTConfig(**args))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    sd = {k: v for k, v in sd.items() if not k.endswith(".attn.bias")}  # old causal-mask buffers
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"transformer.wte.weight"}, missing  # tied
    return m.to(device).eval()


def random_init_model(seed: int = 0, device: str | None = None) -> GPT:
    """Untrained same-architecture baseline (nanoGPT init: N(0,0.02), scaled c_proj)."""
    torch.manual_seed(seed)
    c = GPTConfig()
    m = GPT(c)
    for n, p in m.named_parameters():
        if p.dim() >= 2:
            std = 0.02 / math.sqrt(2 * c.n_layer) if n.endswith("c_proj.weight") else 0.02
            nn.init.normal_(p, 0.0, std)
    return m.to(device or pick_device()).eval()


def set_dropout(m: GPT, p: float):
    m.config.dropout = p
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = p


def save_model(m: GPT, path: str | Path, extra: dict | None = None):
    torch.save({"model": m.state_dict(), "model_args": vars(m.config), **(extra or {})}, path)
