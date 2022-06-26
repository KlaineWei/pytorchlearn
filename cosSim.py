import torch

input_1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
input_2 = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
print(input_1.unsqueeze(1).size())
print(input_2.unsqueeze(0).size())
output = cos(input_1.unsqueeze(1), input_2.unsqueeze(0))
print(output)
print(output.size())
