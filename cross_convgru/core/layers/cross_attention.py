import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim):
#         super(CrossAttention, self).__init__()
#         self.query_dim = query_dim
#         self.context_dim = context_dim
#
#         self.linear_q = nn.Linear(query_dim, query_dim)
#         self.linear_c = nn.Linear(context_dim, query_dim)
#
#     def forward(self, query, context):
#         # Query和Context的维度分别为 [batch_size, query_len, query_dim] 和 [batch_size, context_len, context_dim]
#         # 首先将Query和Context分别通过线性变换
#         query_proj = self.linear_q(query)  # [batch_size, query_len, query_dim]
#         context_proj = self.linear_c(context)  # [batch_size, context_len, query_dim]
#
#         # 计算注意力权重
#         attention_weights = torch.bmm(query_proj, context_proj.transpose(1, 2))  # [batch_size, query_len, context_len]
#         attention_weights = F.softmax(attention_weights, dim=-1)
#
#         # 对Context序列进行加权求和
#         attended_context = torch.bmm(attention_weights, context)  # [batch_size, query_len, context_dim]
#
#         return attended_context, attention_weights

import torch
import torch.nn as nn

class CrossAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention3D, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # query, key, value shape: (B, C, T, H, W)
        B,  C, H, W = query.size()

        # Flatten spatial and temporal dimensions
        query = query.view(B, C,  H * W).permute(2, 0, 1)  # (T*H*W, B, C)
        key = key.view(B, C,  H * W).permute(2, 0, 1)      # (T*H*W, B, C)
        value = value.view(B, C,  H * W).permute(2, 0, 1)  # (T*H*W, B, C)

        attn_output, attn_weights = self.multihead_attn(query, key, value)

        # Reshape back to (B, C, T, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C,  H, W)

        return attn_output, attn_weights

# # 设置参数
# embed_dim = 64
# num_heads = 4
#
# # 创建交叉注意力模块
# cross_attention_3d = CrossAttention3D(embed_dim, num_heads)
#
# # 创建示例输入（query、key、value）
# batch_size = 2
# C = 64  # 通道数
# T = 10  # 时间步数
# H, W = 32, 32  # 高度和宽度
# query = torch.randn(batch_size, C, T, H, W)
# key = torch.randn(batch_size, C, T, H, W)
# value = torch.randn(batch_size, C, T, H, W)
#
# # 前向传播
# attn_output, attn_weights = cross_attention_3d(query, key, value)
#
# print("Attention output shape:", attn_output.shape)
# print("Attention weights shape:", attn_weights.shape)
