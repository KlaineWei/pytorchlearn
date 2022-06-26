import torch

z1 = torch.randn(256, 3200)
z2 = torch.randn(256, 3200)
sim11 = torch.matmul(z1, z1.T)
sim22 = torch.matmul(z2, z2.T)
sim12 = torch.matmul(z1, z2.T)
print(sim11.size())
raw_score1 = torch.cat([sim12, sim11], dim=-1)
print(raw_score1.size())
raw_score2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
print(raw_score2.size())
logists = torch.cat([raw_score1, raw_score2], dim=-2)
print(logists.size())
