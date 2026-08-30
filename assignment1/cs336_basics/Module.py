import torch
import math
from torch import nn
from einops import einsum, rearrange
from collections.abc import Callable, Iterable
from typing import Optional


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        root_mean_square = (1 / self.d_model * einsum(x, x, "... seq_len d_in, ... seq_len d_in -> ... seq_len") + self.eps).sqrt()
        x = x / root_mean_square.unsqueeze(-1)
        x = einsum(x, self.weight, "... seq_len d_model, d_model -> ... seq_len d_model")
        return x.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu_layer = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = self.silu_layer(x1)
        x3 = self.w3(x)
        x2 = einsum(x1, x3, "... d_ff, ... d_ff -> ... d_ff")
        x2 = self.w2(x2)
        return x2

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(ROPE, self).__init__()
        token_position = torch.linspace(0, max_seq_len - 1, steps=max_seq_len)  # The i-th position
        pair_index = torch.arange(1, d_k // 2 + 1)

        token_position = rearrange(token_position, "seq_len -> seq_len 1")
        pair_index = rearrange(pair_index, "d_k -> 1 d_k")
        angle = token_position / (theta ** ((2 * pair_index - 2) / d_k))
        cos_table = angle.cos()  # (seq_len, d_k/2)
        sin_table = angle.sin()  # (seq_len, d_k/2)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... seq_len (d_k_over_two two) -> ... seq_len d_k_over_two two", two=2)
        cos_values = self.cos_table[token_positions]  # (seq_len, d_k/2)
        sin_values = self.sin_table[token_positions]  # (seq_len, d_k/2)
        new0 = x[..., 0] * cos_values - x[..., 1] * sin_values  # (seq_len, d_k/2)
        new1 = x[..., 0] * sin_values + x[..., 1] * cos_values  # (seq_len, d_k/2)
        x[..., 0], x[..., 1] = new0, new1
        x = rearrange(x, "... seq_len d_k_over_two two -> ... seq_len (d_k_over_two two)")
        return x

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = x.max(self.dim, keepdim=True).values
        x = x - x_max
        exp_x = x.exp()
        return exp_x / exp_x.sum(self.dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask):
        d_k = K.shape[-1]
        QK = einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k")
        QK /= math.sqrt(d_k)
        QK = QK.masked_fill(~mask, -torch.inf)
        QK = self.softmax(QK)
        QKV = einsum(QK, V, "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v")
        return QKV

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, positional_encoding=None):
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attn = ScaledDotProductAttention()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if positional_encoding is not None:
            self.pe = ROPE(
                positional_encoding["theta"],
                d_model / num_heads,
                positional_encoding["max_seq_len"]
            )
        else:
            self.pe = None


    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq_len (num_heads d_heads) -> ... num_heads seq_len d_heads", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_heads) -> ... num_heads seq_len d_heads", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_heads) -> ... num_heads seq_len d_heads", num_heads=self.num_heads)
        seq_len = Q.shape[-2]
        if self.pe is not None:
            Q = self.pe(Q, torch.arange(seq_len))
            K = self.pe(K, torch.arange(seq_len))
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        attn_output = rearrange(self.attn(Q, K, V, mask), "... num_heads seq_len d_heads -> ... seq_len (num_heads d_heads)", num_heads=self.num_heads)
        attn_output = self.output_proj(attn_output)
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000, max_seq_len=2048):
        super(TransformerBlock, self).__init__()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, {
            "theta": theta,
            "max_seq_len": max_seq_len
        })
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta=10000):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, targets):
        target_logit = x[torch.arange(x.size(0)), targets]
        max_values = x.max(-1).values
        lse = max_values + (x - max_values.unsqueeze(-1)).exp().sum(-1).log()
        return (-target_logit + lse).mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("first_moment", 0)
                v = state.get("second_moment", 0)
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                grad = p.grad.data
                p.data -= weight_decay * lr * p.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                p.data -= lr_t * m / (v.sqrt() + eps)
                state["first_moment"] = m
                state["second_moment"] = v
                state["t"] = t + 1

        return loss