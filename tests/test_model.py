import torch
from model import RMSNorm

def test_rmsnorm():
    norm = RMSNorm(384)
    x = torch.randn(2, 10, 384)
    out = norm(x)
    assert out.shape == x.shape
