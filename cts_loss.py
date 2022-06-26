import torch


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


batch_size = 256
D = 3200
N = 2 * batch_size

z_i = torch.randn([batch_size, D])
z_j = torch.randn([batch_size, D])
z = torch.cat((z_i, z_j), dim=0)  # 2B * D
print("z_i", z_i.size())
print("z_j", z_j.size())
print("z", z.size())

temp = 0.07

sim = torch.mm(z, z.T) / temp  # 2B * 2B
print("sim", sim.size())

sim_i_j = torch.diag(sim, batch_size)  # B*1
sim_j_i = torch.diag(sim, -batch_size)  # B*1
print("sim_i_j", sim_i_j.size())
print("sim_j_i", sim_j_i.size())

positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
mask = mask_correlated_samples(batch_size)
negative_samples = sim[mask].reshape(N, -1)
print("pos", positive_samples.size())
print("mask", mask.size())
print("neg", negative_samples.size())

labels = torch.zeros(N).to(positive_samples.device).long()
logits = torch.cat((positive_samples, negative_samples), dim=1)
