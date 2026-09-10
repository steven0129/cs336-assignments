import torch
from cs336_basics.Module import Embedding, TransformerBlock

total_size_bytes_before = 0
total_size_bytes_after = 0

def pack_before_hook(t):
    if isinstance(t, torch.nn.Parameter):
        return t

    global total_size_bytes_before
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes_before += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def pack_after_hook(t):
    if isinstance(t, torch.nn.Parameter):
        return t

    global total_size_bytes_after
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes_after += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


if __name__ == "__main__":
    d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length)
    if torch.cuda.is_available():
        block = block.to('cuda')

    x1 = torch.randn((4, context_length, d_model), requires_grad=True)
    if torch.cuda.is_available():
        x1 = x1.to('cuda')
    with torch.autograd.graph.saved_tensors_hooks(pack_before_hook, unpack_hook):
        y = block(x1)

    print(f"Total size before compilation: {total_size_bytes_before / (1024 ** 2):.2f} MB")
    print()

    block = torch.compile(block, fullgraph=True)
    x2 = torch.randn((4, context_length, d_model), requires_grad=True)
    if torch.cuda.is_available():
        x2 = x2.to('cuda')
    with torch.autograd.graph.saved_tensors_hooks(pack_after_hook, unpack_hook):
        y = block(x2)

    print(f"Total size after compilation: {total_size_bytes_after / (1024 ** 2):.2f} MB")
    print()