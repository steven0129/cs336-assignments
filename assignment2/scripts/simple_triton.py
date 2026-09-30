import torch
import triton
import triton.language as tl


@triton.jit
def matmul_forward(
    a_ptr,
    b_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr = tl.float32
):
    block_id = tl.program_id(0)
    num_blocks_each_row = tl.cdiv(N, BLOCK_SIZE_N)
    block_id_m = block_id // num_blocks_each_row
    block_id_n = block_id % num_blocks_each_row

    offsets_m = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offsets_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offsets_m[:, None] * K + offsets_k[None, :]
        b_ptrs = b_ptr + offsets_k[:, None] * N + offsets_n[None, :]
        mask_a = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        mask_b = (offsets_k[:, None] < K) & (offsets_n[None, :] < N)

        A = tl.load(a_ptrs, mask=mask_a, other=0)
        B = tl.load(b_ptrs, mask=mask_b, other=0)
        acc += tl.dot(A, B)

    output_ptrs = output_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    mask_output = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask_output)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    device = a.device
    dtype = a.dtype

    c = torch.zeros((M, N), device=device, dtype=dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) *
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    matmul_forward[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )

    return c


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda', dtype=torch.float32)
    b = torch.tensor([[1, 2], [3, 4], [5, 6]], device='cuda', dtype=torch.float32)
    print(matmul(a, b))
    print(torch.matmul(a, b))
