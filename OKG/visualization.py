import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize(model, dataset, triple_idx=None):
    if triple_idx is None:
        triple_idx = np.random.randint(len(dataset.triples))
    
    triple = dataset.triples[triple_idx]
    head, rel, tail = triple
    
    # 获取实体和关系的ID
    head_id = dataset.entity2id[head]
    rel_id = dataset.relation2id[rel]
    tail_id = dataset.entity2id[tail]
    
    # 获取嵌入向量
    head_embed = model.entity_embedding.weight.data[head_id].detach().numpy()
    rel_embed = model.relation_embedding.weight.data[rel_id].detach().numpy()
    tail_embed = model.entity_embedding.weight.data[tail_id].detach().numpy()
    
    embeddings = np.vstack([head_embed, rel_embed, tail_embed])
    
    # t-SNE降维，使用较小的perplexity值
    tsne = TSNE(n_components=2, perplexity=1.5, random_state=42, method='exact')
    tsne_result = tsne.fit_transform(embeddings)
    
    # 创建图形
    plt.figure(figsize=(5, 4))
    
    # 获取2D坐标
    head_2d = tsne_result[0]
    rel_2d = tsne_result[1]
    tail_2d = tsne_result[2]
    predicted_2d = head_2d + rel_2d
    
    # 绘制向量
    plt.quiver(0, 0, head_2d[0], head_2d[1], angles='xy', scale_units='xy', scale=1, color='b', label='Head')
    plt.quiver(head_2d[0], head_2d[1], rel_2d[0], rel_2d[1], angles='xy', scale_units='xy', scale=1, color='r', label='Relation')
    plt.quiver(0, 0, predicted_2d[0], predicted_2d[1], angles='xy', scale_units='xy', scale=1, color='g', label='Head + Relation')
    plt.quiver(0, 0, tail_2d[0], tail_2d[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Tail')
    
    # 添加点
    plt.scatter(0, 0, c='black', s=100, label='Origin')
    plt.scatter(head_2d[0], head_2d[1], c='blue', s=100)
    plt.scatter(predicted_2d[0], predicted_2d[1], c='green', s=100)
    plt.scatter(tail_2d[0], tail_2d[1], c='purple', s=100)
    
    # 添加文本标签
    plt.annotate(head, (head_2d[0], head_2d[1]), xytext=(10, 10), textcoords='offset points')
    plt.annotate(rel, (head_2d[0] + rel_2d[0]/2, head_2d[1] + rel_2d[1]/2), xytext=(10, 10), textcoords='offset points')
    plt.annotate(tail, (tail_2d[0], tail_2d[1]), xytext=(10, 10), textcoords='offset points')
    
    plt.title(f't-SNE Visualization\n{head} + {rel} ≈ {tail}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 计算并显示度量
    pred_tail_dist = np.linalg.norm(predicted_2d - tail_2d)
    head_tail_dist = np.linalg.norm(head_2d - tail_2d)
    rel_magnitude = np.linalg.norm(rel_2d)
    
    print(f'预测点到尾实体的距离: {pred_tail_dist:.4f}')
    print(f'头实体到尾实体的距离: {head_tail_dist:.4f}')
    print(f'关系向量长度: {rel_magnitude:.4f}')