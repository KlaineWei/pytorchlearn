import torch

x = torch.randn(3, 3)
y = torch.zeros_like(x)
z = torch.ones_like(x)
print(x)
print(y)
print(z)
print(torch.where(x > 0.0, y, z))
