import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    off = pid * 1024
    x = tl.load(X + off)
    y = tl.load(Y + off)
    tl.store(Z + off, x + y)

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
z = torch.empty_like(x)

add_kernel[(1,)](x, y, z, 1024)
print("Triton kernel ran successfully.")
