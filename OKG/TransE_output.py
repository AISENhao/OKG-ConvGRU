import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text
from KG import spatial_triples, temporal_triples  # 空间三元组和时间三元组
from TransE_main import model, dataset
import sys
sys.path.append('../')
from model import train_transe, train_transh

# 假设 spatial_triples
transe_model, transe_dataset = model,dataset

# 输出实体嵌入向量
entity_embeddings = transe_model.entity_embedding.weight.data.numpy()
print("Entity Embeddings:")
print(entity_embeddings)

# 输出关系嵌入向量
relation_embeddings = transe_model.relation_embedding.weight.data.numpy()
print("Relation Embeddings:")
print(relation_embeddings)


# 可视化所有实体和关系嵌入向量
def visualize_embeddings(entity_embeddings, relation_embeddings, transe_dataset, save_path=None, dpi=400):
    entity_vectors = entity_embeddings
    relation_vectors = relation_embeddings
    entity_names = transe_dataset.entities
    relation_names = transe_dataset.relations

    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    all_vectors = np.concatenate([entity_vectors, relation_vectors], axis=0)
    vectors_2d = tsne.fit_transform(all_vectors)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:len(entity_vectors), 0], vectors_2d[:len(entity_vectors), 1], c='blue', alpha=0.6,
                label='Entities')
    plt.scatter(vectors_2d[len(entity_vectors):, 0], vectors_2d[len(entity_vectors):, 1], c='red', alpha=0.6,
                label='Relations')

    # 创建注解文本列表
    texts = []
    for i, name in enumerate(entity_names):
        texts.append(plt.annotate(name, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=8, ha='center', va='bottom',
                                  color='blue'))

    for i, name in enumerate(relation_names):
        texts.append(
            plt.annotate(name, (vectors_2d[len(entity_vectors) + i, 0], vectors_2d[len(entity_vectors) + i, 1]),
                         fontsize=8,
                         ha='center', va='bottom', color='red'))

    # 调整注解文本以避免重叠
    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})

    plt.title('Temporal unit entity and relation feature vector')
    plt.legend()

    # 如果指定了保存路径，则保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


# 指定保存路径
save_path = 'filename.png'

# 调用可视化函数
visualize_embeddings(entity_embeddings, relation_embeddings, transe_dataset, save_path=save_path)