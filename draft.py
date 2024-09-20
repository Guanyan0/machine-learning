import torch
import torch.nn as nn

# 定义输入序列
embed_size = 512
num_heads = 8
seq_length = 10
batch_size = 32

# 模拟输入序列
query = torch.randn(seq_length, batch_size, embed_size)
key = torch.randn(seq_length, batch_size, embed_size)
value = torch.randn(seq_length, batch_size, embed_size)

# 定义多头注意力机制
multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)

# 计算多头注意力的输出
attn_output, attn_weights = multihead_attention(query, key, value)

print("Attention Output Shape:", attn_output.shape)
print("Attention Weights Shape:", attn_weights.shape)
