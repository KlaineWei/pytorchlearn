import torch
from info_nce import InfoNCE

# loss = InfoNCE()
# batch_size, embedding_size = 32, 128
# query = torch.randn(batch_size, embedding_size)
# positive_key = torch.randn(batch_size, embedding_size)
# output = loss(query, positive_key)
# print(query)
# print(positive_key)
# print(output)

# loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
# batch_size, num_negative, embedding_size = 32, 48, 128
# query = torch.randn(batch_size, embedding_size)
# positive_key = torch.randn(batch_size, embedding_size)
# negative_keys = torch.randn(num_negative, embedding_size)
# output = loss(query, positive_key, negative_keys)

loss = InfoNCE(negative_mode='paired')
batch_size, num_negative, embedding_size = 32, 6, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)
print(query)
print(positive_key)
print(negative_keys)
print(output)
