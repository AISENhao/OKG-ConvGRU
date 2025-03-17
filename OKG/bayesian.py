from KG import spatial_triples, temporal_triples  # 空间三元组和时间三元组
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from evaluate import evaluate_transhmodel, filtered_ranking
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
class KGDataset(Dataset):
    def __init__(self, triples, num_negative=24):
        self.triples = triples
        self.num_negative = num_negative

        # 创建实体和关系字典
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

        # 生成负样本
        neg_heads = []
        neg_rels = []
        neg_tails = []

        for _ in range(self.num_negative):
            if np.random.random() < 0.5:
                # 破坏头实体
                neg_head = np.random.randint(self.num_entities)
                while neg_head == head_id:
                    neg_head = np.random.randint(self.num_entities)
                neg_heads.append(neg_head)
                neg_rels.append(rel_id)
                neg_tails.append(tail_id)
            else:
                # 破坏尾实体
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
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0, C=0.1):
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


def train_transh(triples, num_negative, epochs=100, batch_size=12, embedding_dim=100, lr=0.01, margin=1.0, C=0.1):
    dataset = KGDataset(triples, num_negative)
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

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

    return model, dataset


def calculate_mrr(model, dataset, test_triples=None):
    """
    计算过滤后的排名（去除训练集中的真实三元组）
    """
    # 构建所有已知的真实三元组集合
    all_true_triples = set()
    for h, r, t in dataset.triples:
        all_true_triples.add((dataset.entity2id[h], dataset.relation2id[r], dataset.entity2id[t]))

    if test_triples is None:
        test_triples = dataset.triples

    filtered_hits = {1: [], 3: [], 10: []}
    filtered_mean_rank = []
    filtered_mean_reciprocal_rank = []

    model.eval()
    with torch.no_grad():
        for triple in test_triples:
            head_id = dataset.entity2id[triple[0]]
            rel_id = dataset.relation2id[triple[1]]
            tail_id = dataset.entity2id[triple[2]]

            # Filtered head prediction
            scores_head = []
            for ent_id in range(dataset.num_entities):
                if (ent_id, rel_id, tail_id) in all_true_triples and ent_id != head_id:
                    continue
                h_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([tail_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_head.append((score.item(), ent_id))

            scores_head.sort()
            filtered_rank_head = 1
            for score, ent_id in scores_head:
                if ent_id == head_id:
                    break
                filtered_rank_head += 1

            # Filtered tail prediction
            scores_tail = []
            for ent_id in range(dataset.num_entities):
                if (head_id, rel_id, ent_id) in all_true_triples and ent_id != tail_id:
                    continue
                h_emb = model.entity_embedding(torch.LongTensor([head_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_tail.append((score.item(), ent_id))

            scores_tail.sort()
            filtered_rank_tail = 1
            for score, ent_id in scores_tail:
                if ent_id == tail_id:
                    break
                filtered_rank_tail += 1

            # 更新指标
            for rank in [filtered_rank_head, filtered_rank_tail]:
                filtered_mean_rank.append(rank)
                filtered_mean_reciprocal_rank.append(1.0 / rank)
                for k in [1, 3, 10]:
                    filtered_hits[k].append(1 if rank <= k else 0)

    return sum(filtered_mean_reciprocal_rank) / len(filtered_mean_reciprocal_rank)

def calculate_hits_at_10(model, dataset, test_triples=None):
    """
    计算过滤后的排名（去除训练集中的真实三元组）
    """
    # 构建所有已知的真实三元组集合
    all_true_triples = set()
    for h, r, t in dataset.triples:
        all_true_triples.add((dataset.entity2id[h], dataset.relation2id[r], dataset.entity2id[t]))

    if test_triples is None:
        test_triples = dataset.triples

    filtered_hits = {1: [], 3: [], 10: []}
    filtered_mean_rank = []
    filtered_mean_reciprocal_rank = []

    model.eval()
    with torch.no_grad():
        for triple in test_triples:
            head_id = dataset.entity2id[triple[0]]
            rel_id = dataset.relation2id[triple[1]]
            tail_id = dataset.entity2id[triple[2]]

            # Filtered head prediction
            scores_head = []
            for ent_id in range(dataset.num_entities):
                if (ent_id, rel_id, tail_id) in all_true_triples and ent_id != head_id:
                    continue
                h_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([tail_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_head.append((score.item(), ent_id))

            scores_head.sort()
            filtered_rank_head = 1
            for score, ent_id in scores_head:
                if ent_id == head_id:
                    break
                filtered_rank_head += 1

            # Filtered tail prediction
            scores_tail = []
            for ent_id in range(dataset.num_entities):
                if (head_id, rel_id, ent_id) in all_true_triples and ent_id != tail_id:
                    continue
                h_emb = model.entity_embedding(torch.LongTensor([head_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_tail.append((score.item(), ent_id))

            scores_tail.sort()
            filtered_rank_tail = 1
            for score, ent_id in scores_tail:
                if ent_id == tail_id:
                    break
                filtered_rank_tail += 1

            # 更新指标
            for rank in [filtered_rank_head, filtered_rank_tail]:
                filtered_mean_rank.append(rank)
                filtered_mean_reciprocal_rank.append(1.0 / rank)
                for k in [1, 3, 10]:
                    filtered_hits[k].append(1 if rank <= k else 0)


    return sum(filtered_hits[10]) / len(filtered_hits[10])
def evaluate_transhmodel(model, dataset, metric='MRR'):
    """评估模型的性能"""
    if metric == 'MRR':
        # 计算 MRR
        mrr = calculate_mrr(model, dataset)
        return mrr
    elif metric == 'Hits@10':
        # 计算 Hits@10
        hits_at_10 = calculate_hits_at_10(model, dataset)
        return hits_at_10
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def evaluate_transh_mrr(model, dataset):
    """评估模型的MRR"""
    mrr = evaluate_transhmodel(model, dataset, metric='MRR')
    return -mrr  # 我们需要最小化这个值，所以返回负的MRR

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
            # 头实体投影加关系到尾实体投影的线
            plt.plot([head_minus_w_r_plus_r_2d[i, 0], tail_minus_w_r_2d[i, 0]],
                     [head_minus_w_r_plus_r_2d[i, 1], tail_minus_w_r_2d[i, 1]],
                     'gray', alpha=0.2)

        plt.legend()
        plt.title('TransH Embeddings Visualization')
        plt.show()

        # 计算平均距离误差
        avg_distance = np.mean([
            np.linalg.norm(head_minus_w_r_plus_r_2d[i] - tail_minus_w_r_2d[i])
            for i in range(num_samples)
        ])
        print(f"Average distance between ((h - w_r) + r) and (t - w_r) in 2D space: {avg_distance:.4f}")


# 定义搜索空间
space = [Integer(1, 100, name='num_negative')]

# 定义目标函数
@use_named_args(space)
def objective(num_negative):
    model, dataset = train_transh(temporal_triples, num_negative=num_negative)
    mrr = evaluate_transh_mrr(model, dataset)
    return mrr


# 运行贝叶斯优化
res = gp_minimize(objective, space, n_calls=20, random_state=42)

# 打印最佳参数
print(f"Best number of negative samples: {res.x[0]}")
print(f"Best MRR: {-res.fun:.4f}")

# 使用最佳参数重新训练模型
best_model, best_dataset = train_transh(spatial_triples, num_negative=res.x[0])

# 评估最佳模型
evaluate_transhmodel(best_model, best_dataset, metric='MRR')

# 可视化
visualize_transh_triples(best_model, best_dataset, num_samples=50)

# 绘制优化过程
plt.figure(figsize=(10, 6))
plot_convergence(res)
plt.title('Bayesian Optimization Convergence')
plt.show()