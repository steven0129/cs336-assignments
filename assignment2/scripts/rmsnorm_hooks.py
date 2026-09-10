import torch
from cs336_basics.Module import RMSNorm


def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


if __name__ == "__main__":
    x = torch.randn((4, 512, 2560), requires_grad=True)
    rmsnorm = RMSNorm(x.shape[-1])

    print('Before compilation....')
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = rmsnorm(x)
        y.sum().backward()

    print()
    print('After compilation....')
    rmsnorm = torch.compile(RMSNorm(x.shape[-1]))
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = rmsnorm(x)
        y.sum().backward()