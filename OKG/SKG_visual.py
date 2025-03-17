import sys
sys.path.append('../')
from KG import spatial_triples, temporal_triples, myself_triples  # 空间三元组和时间三元组
from model import train_transe, train_transh
from evaluate import evaluate_transhmodel, evaluate_tranemodel

import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
G = nx.DiGraph()

# 定义特殊节点列表
special_nodes = ['Chl-a', 'SST', 'PAR', 'POC', 'PIC', 'NFLH']

# 解析三元组数据并添加到图中
triples =myself_triples
# 添加边
for head, relation, tail in triples:
    G.add_edge(head, tail, relation=relation)

# 设置绘图参数
plt.figure(figsize=(15, 15), dpi=400)  # 设置图形尺寸和DPI

# 使用spring_layout但增加k值使节点更分散
pos = nx.spring_layout(G, k=1, iterations=50)  # 增加k值和迭代次数

# 准备节点颜色列表
node_colors = ['pink' if node in special_nodes else 'skyblue' for node in G.nodes()]

# 绘制节点（添加边界线）
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=4000,  # 略微增大节点尺寸
                       alpha=0.7,
                       edgecolors='black',  # 添加黑色边界线
                       linewidths=1.5)  # 设置边界线宽度

# 绘制边
nx.draw_networkx_edges(G, pos,
                        edge_color='gray',
                        arrows=True,
                        arrowsize=20,
                        width=1.5)  # 增加边的宽度

# 添加节点标签
nx.draw_networkx_labels(G, pos, font_size=12)  # 增大字体

# 添加边标签
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

# # 添加图例
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='pink', edgecolor='black', alpha=0.7, label='Special Variables'),
#     Patch(facecolor='skyblue', edgecolor='black', alpha=0.7, label='Other Entities')
# ]
# plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# 调整布局
plt.axis('off')
plt.tight_layout()

# 保存图像到指定路径
save_path = "SKG_whole.png"  # 替换为你的保存路径
plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 保存图像，设置DPI和边界

# # 显示图形
# plt.show()
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 创建有向图
# G = nx.DiGraph()
#
# # 定义特殊节点列表
# special_nodes = ['chlorophyll_a', 'SST', 'PAR', 'POC', 'PIC', 'NFLH']
#
# # 解析三元组数据并添加到图中
# triples = temporal_triples
# # 添加边
# for head, relation, tail in triples:
#     G.add_edge(head, tail, relation=relation)
#
# # 设置绘图参数
# plt.figure(figsize=(24, 24))  # 增大图形尺寸
#
# # 使用spring_layout但增加k值使节点更分散
# pos = nx.spring_layout(G, k=2, iterations=100)  # 增加k值和迭代次数
#
# # 准备节点颜色列表
# node_colors = ['pink' if node in special_nodes else 'skyblue' for node in G.nodes()]
#
# # 绘制节点（添加边界线）
# nx.draw_networkx_nodes(G, pos,
#                       node_color=node_colors,
#                       node_size=2600,  # 略微增大节点尺寸
#                       alpha=0.7,
#                       edgecolors='black',  # 添加黑色边界线
#                       linewidths=1.5)  # 设置边界线宽度
#
# # 绘制边
# nx.draw_networkx_edges(G, pos,
#                       edge_color='gray',
#                       arrows=True,
#                       arrowsize=20,
#                       width=1.5)  # 增加边的宽度
#
# # 添加节点标签
# nx.draw_networkx_labels(G, pos, font_size=12)  # 增大字体
#
# # 添加边标签
# edge_labels = nx.get_edge_attributes(G, 'relation')
# nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
#
# # 添加图例
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Special Variables'),
#     Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='Other Entities')
# ]
# plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
#
# # 调整布局
# plt.axis('off')
# plt.tight_layout()
#
# # 显示图形
# plt.show()