from KG import spatial_triples, temporal_triples,myself_triples  #空间三元组和时间三元组
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from evaluate import evaluate_transhmodel, filtered_ranking

class KGDataset(Dataset):
    def __init__(self, triples, num_negative=30):
        self.triples = triples
        self.num_negative = num_negative

        # Create entity and relation dictionaries
        self.entities = sorted(list(set([t[0] for t in triples] + [t[2] for t in triples])))
        self.relations = sorted(list(set([t[1] for t in triples])))

        self.entity2id = {ent: idx for idx, ent in enumerate(self.entities)}
        self.relation2id = {rel: idx for idx, rel in enumerate(self.relations)}

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        head_id = self.entity2id[pos_triple[0]]
        rel_id = self.relation2id[pos_triple[1]]
        tail_id = self.entity2id[pos_triple[2]]

        # Generate negative samples by corrupting head or tail
        neg_heads = []
        neg_rels = []
        neg_tails = []

        for _ in range(self.num_negative):
            if np.random.random() < 0.5:
                # Corrupt head
                neg_head = np.random.randint(self.num_entities)
                while neg_head == head_id:
                    neg_head = np.random.randint(self.num_entities)
                neg_heads.append(neg_head)
                neg_rels.append(rel_id)
                neg_tails.append(tail_id)
            else:
                # Corrupt tail
                neg_tail = np.random.randint(self.num_entities)
                while neg_tail == tail_id:
                    neg_tail = np.random.randint(self.num_entities)
                neg_heads.append(head_id)
                neg_rels.append(rel_id)
                neg_tails.append(neg_tail)

        return (torch.LongTensor([head_id]),
                torch.LongTensor([rel_id]),
                torch.LongTensor([tail_id]),
                torch.LongTensor(neg_heads),
                torch.LongTensor(neg_rels),
                torch.LongTensor(neg_tails))


class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=120, margin=1.0, C=0.1):
        super(TransH, self).__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.C = C  # 软约束权重

        # 实体embedding
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        # 关系embedding
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        # 关系超平面的法向量
        self.normal_vector = nn.Embedding(num_relations, embedding_dim)

        # 初始化embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.normal_vector.weight.data)

        # 归一化实体embeddings
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)

    def _transfer(self, e, norm):
        """将实体投影到关系特定的超平面上"""
        norm = F.normalize(norm, p=2, dim=1)
        return e - torch.sum(e * norm, dim=1, keepdim=True) * norm

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # 获取embeddings
        pos_h_emb = self.entity_embedding(pos_h)
        pos_t_emb = self.entity_embedding(pos_t)
        pos_r_emb = self.relation_embedding(pos_r)
        pos_norm = self.normal_vector(pos_r)

        neg_h_emb = self.entity_embedding(neg_h)
        neg_t_emb = self.entity_embedding(neg_t)
        neg_r_emb = self.relation_embedding(neg_r)
        neg_norm = self.normal_vector(neg_r)

        # 投影到超平面
        pos_h_perp = self._transfer(pos_h_emb, pos_norm)
        pos_t_perp = self._transfer(pos_t_emb, pos_norm)
        neg_h_perp = self._transfer(neg_h_emb, neg_norm)
        neg_t_perp = self._transfer(neg_t_emb, neg_norm)

        # 计算得分
        pos_score = torch.norm(pos_h_perp + pos_r_emb - pos_t_perp, p=2, dim=1)
        neg_score = torch.norm(neg_h_perp + neg_r_emb - neg_t_perp, p=2, dim=1)

        # 基本损失
        basic_loss = torch.mean(F.relu(self.margin + pos_score - neg_score))

        # 软约束：保持法向量和关系向量正交
        r_norm = torch.norm(self.relation_embedding.weight.data, p=2, dim=1)
        w_norm = torch.norm(self.normal_vector.weight.data, p=2, dim=1)

        orthogonal_constraint = torch.sum(
            torch.abs(torch.sum(
                self.relation_embedding.weight.data * self.normal_vector.weight.data,
                dim=1
            ) / (r_norm * w_norm))
        )

        # 总损失
        loss = basic_loss + self.C * orthogonal_constraint
        return loss

    def normalize_embeddings(self):
        """归一化embeddings"""
        with torch.no_grad():
            self.entity_embedding.weight.data = F.normalize(
                self.entity_embedding.weight.data, p=2, dim=1
            )
            self.relation_embedding.weight.data = F.normalize(
                self.relation_embedding.weight.data, p=2, dim=1
            )
            self.normal_vector.weight.data = F.normalize(
                self.normal_vector.weight.data, p=2, dim=1
            )


def train_transh(triples, epochs=500, batch_size=32, embedding_dim=100, lr=0.01, margin=1.0, C=0.05):
    dataset = KGDataset(triples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransH(dataset.num_entities, dataset.num_relations,
                   embedding_dim=embedding_dim, margin=margin, C=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for pos_h, pos_r, pos_t, neg_h, neg_r, neg_t in dataloader:
            loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 归一化embeddings
            model.normalize_embeddings()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

    return model, dataset

model, dataset = train_transh(temporal_triples)



import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch


def visualize_transh_triples(model, dataset, num_samples=100):
    """
    可视化 TransH 三元组关系: (h - w_r) + r ≈ (t - w_r)
    """
    model.eval()
    with torch.no_grad():
        # 随机采样一些三元组
        all_triples = list(dataset.triples)
        sampled_indices = np.random.choice(len(all_triples), num_samples, replace=False)
        sampled_triples = [all_triples[i] for i in sampled_indices]

        # 收集向量
        heads = []
        relations = []
        tails = []
        head_minus_w_r = []  # h - w_r 的结果
        tail_minus_w_r = []  # t - w_r 的结果
        head_minus_w_r_plus_r = []  # (h - w_r) + r 的结果

        for h, r, t in sampled_triples:
            # 获取embeddings
            h_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[h]])).numpy()
            r_emb = model.relation_embedding(torch.LongTensor([dataset.relation2id[r]])).numpy()
            t_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[t]])).numpy()
            w_r_emb = model.normal_vector(torch.LongTensor([dataset.relation2id[r]])).numpy()

            # 计算投影
            h_proj = model._transfer(torch.tensor(h_emb), torch.tensor(w_r_emb)).numpy()
            t_proj = model._transfer(torch.tensor(t_emb), torch.tensor(w_r_emb)).numpy()

            heads.append(h_emb[0])
            relations.append(r_emb[0])
            tails.append(t_emb[0])
            head_minus_w_r.append(h_proj[0])
            tail_minus_w_r.append(t_proj[0])
            head_minus_w_r_plus_r.append(h_proj[0] + r_emb[0])

        # 将所有向量堆叠在一起进行降维
        all_vectors = np.vstack([
            heads,
            relations,
            tails,
            head_minus_w_r,
            tail_minus_w_r,
            head_minus_w_r_plus_r
        ])

        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(all_vectors)

        # 分离降维后的结果
        heads_2d = vectors_2d[:num_samples]
        relations_2d = vectors_2d[num_samples:2 * num_samples]
        tails_2d = vectors_2d[2 * num_samples:3 * num_samples]
        head_minus_w_r_2d = vectors_2d[3 * num_samples:4 * num_samples]
        tail_minus_w_r_2d = vectors_2d[4 * num_samples:5 * num_samples]
        head_minus_w_r_plus_r_2d = vectors_2d[5 * num_samples:]

        # 绘图
        plt.figure(figsize=(12, 8))

        # 绘制实体和关系点
        plt.scatter(heads_2d[:, 0], heads_2d[:, 1], c='blue', label='Head', alpha=0.6)
        plt.scatter(relations_2d[:, 0], relations_2d[:, 1], c='green', label='Relation', alpha=0.6)
        plt.scatter(tails_2d[:, 0], tails_2d[:, 1], c='red', label='Tail', alpha=0.6)
        plt.scatter(head_minus_w_r_2d[:, 0], head_minus_w_r_2d[:, 1], c='cyan', label='Head - w_r', alpha=0.6)
        plt.scatter(tail_minus_w_r_2d[:, 0], tail_minus_w_r_2d[:, 1], c='magenta', label='Tail - w_r', alpha=0.6)
        plt.scatter(head_minus_w_r_plus_r_2d[:, 0], head_minus_w_r_plus_r_2d[:, 1], c='purple',
                    label='(Head - w_r) + Relation', alpha=0.6)

        # 为每个三元组绘制连接线
        for i in range(num_samples):
            # 头实体到头实体投影的线
            plt.plot([heads_2d[i, 0], head_minus_w_r_2d[i, 0]],
                     [heads_2d[i, 1], head_minus_w_r_2d[i, 1]],
                     'gray', alpha=0.2)
            # 头实体到头实体投影的线
            plt.plot([tails_2d[i, 0], tail_minus_w_r_2d[i, 0]],
                     [tails_2d[i, 1], tail_minus_w_r_2d[i, 1]],
                     'gray', alpha=0.2)


            # 头实体投影加关系到尾实体投影的线
            plt.plot([head_minus_w_r_plus_r_2d[i, 0], tail_minus_w_r_2d[i, 0]],
                     [head_minus_w_r_plus_r_2d[i, 1], tail_minus_w_r_2d[i, 1]],
                     'orange', alpha=0.5)

        plt.legend()
        # 保存图像到指定路径
        save_path = "TransH_temporal.png"  # 替换为你的保存路径
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 保存图像，设置DPI和边界
        plt.show()

        # 计算平均距离误差
        avg_distance = np.mean([
            np.linalg.norm(head_minus_w_r_plus_r_2d[i] - tail_minus_w_r_2d[i])
            for i in range(num_samples)
        ])
        print(f"Average distance between ((h - w_r) + r) and (t - w_r) in 2D space: {avg_distance:.4f}")


# 在训练后评估模型
evaluate_transhmodel(model, dataset)

visualize_transh_triples(model, dataset, num_samples=50)