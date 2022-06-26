import torch
from torch import nn

z1 = torch.randn([256, 3200])
z2 = torch.randn([256, 3200])
z_negative = torch.randn([256, 3200])
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cos_sim = cos(z1.unsqueeze(1), z2.unsqueeze(0))
labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
print("labels", labels.size())
loss_fct = nn.CrossEntropyLoss()
print("cos_sim", cos_sim.size())
cos_sim_negative = cos(z1.unsqueeze(1), z_negative.unsqueeze(0))
print("cos_sim_negative", cos_sim_negative.size())
cos_sim = torch.cat([cos_sim, cos_sim_negative], 1)
print("cos_sim", cos_sim.size())
weights = torch.where(cos_sim > 0.5, 0, 1)
print("weights", weights.size())
mask_weights = torch.eye(cos_sim.size(0), device=cos_sim.device) - torch.diag_embed(
    torch.diag(weights))
print("mask_weights", mask_weights.size())
weights = weights + torch.cat([mask_weights, torch.zeros_like(cos_sim_negative)], -1)
print("weights", weights.size())
soft_cos_sim = torch.softmax(cos_sim * weights, -1)
print("soft_cos_sim", soft_cos_sim.size())
labels_dis = torch.cat(
            [torch.eye(cos_sim.size(0), device=cos_sim.device)[labels],
             torch.zeros_like(cos_sim_negative)], -1)
print("label_dis", labels_dis.size())
loss = - (labels_dis * torch.log(soft_cos_sim) + (1 - labels_dis) * torch.log(1 - soft_cos_sim))
loss = torch.mean(loss)
print(loss)
