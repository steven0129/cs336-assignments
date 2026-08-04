import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight)

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
        self.weight = nn.Parameter(torch.empty((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        root_mean_square = (1 / self.d_model * einsum(x, x, "... seq_len d_in, ... seq_len d_in -> ... seq_len") + self.eps).sqrt()
        x = x / root_mean_square.unsqueeze(-1)
        x = einsum(x, self.weight, "... seq_len d_model, d_model -> ... seq_len d_model")
        return x.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        x1 = x1 / (1 + torch.exp(-x1))
        x3 = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        x2 = einsum(x1, x3, "... d_ff, ... d_ff -> ... d_ff")
        x2 = einsum(x2, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return x2