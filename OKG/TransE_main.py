from KG import spatial_triples, temporal_triples ,myself_triples #空间三元组和时间三元组
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class KGDataset(Dataset):
    def __init__(self, triples, num_negative=1):
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

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        
        # Normalize entity embeddings
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight.data, p=2, dim=1)
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # Process positive triple
        pos_h_emb = self.entity_embedding(pos_h)
        pos_r_emb = self.relation_embedding(pos_r)
        pos_t_emb = self.entity_embedding(pos_t)
        
        # Process negative triples
        neg_h_emb = self.entity_embedding(neg_h)
        neg_r_emb = self.relation_embedding(neg_r)
        neg_t_emb = self.entity_embedding(neg_t)
        
        # Calculate scores
        pos_score = torch.norm(pos_h_emb + pos_r_emb - pos_t_emb, p=2, dim=1)
        neg_score = torch.norm(neg_h_emb + neg_r_emb - neg_t_emb, p=2, dim=1)
        
        # Calculate loss
        loss = torch.mean(F.relu(self.margin + pos_score - neg_score))
        return loss

def train_transe(triples, epochs=100, batch_size=32, embedding_dim=100, lr=0.01):
    dataset = KGDataset(triples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TransE(dataset.num_entities, dataset.num_relations, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for pos_h, pos_r, pos_t, neg_h, neg_r, neg_t in dataloader:
            loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize entity embeddings after each update
            with torch.no_grad():
                model.entity_embedding.weight.data = F.normalize(
                    model.entity_embedding.weight.data, p=2, dim=1
                )
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset

# Example usage:
model, dataset = train_transe(temporal_triples)


def evaluate_transe(model, dataset, test_triples=None, k_values=[1, 3, 10]):
    """
    评估TransE模型性能
    """
    if test_triples is None:
        test_triples = dataset.triples
    
    hits = {k: [] for k in k_values}
    mean_rank = []
    mean_reciprocal_rank = []
    
    model.eval()
    with torch.no_grad():
        for triple in test_triples:
            head_id = dataset.entity2id[triple[0]]
            rel_id = dataset.relation2id[triple[1]]
            tail_id = dataset.entity2id[triple[2]]
            
            # Head prediction
            scores_head = []
            for ent_id in range(dataset.num_entities):
                h_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([tail_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_head.append((score.item(), ent_id))
            
            # 对分数排序
            scores_head.sort()
            rank_head = 1
            for score, ent_id in scores_head:
                if ent_id == head_id:
                    break
                rank_head += 1
            
            # Tail prediction
            scores_tail = []
            for ent_id in range(dataset.num_entities):
                h_emb = model.entity_embedding(torch.LongTensor([head_id]))
                r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
                t_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                score = torch.norm(h_emb + r_emb - t_emb, p=2)
                scores_tail.append((score.item(), ent_id))
            
            scores_tail.sort()
            rank_tail = 1
            for score, ent_id in scores_tail:
                if ent_id == tail_id:
                    break
                rank_tail += 1
            
            # 计算各项指标
            for rank in [rank_head, rank_tail]:
                mean_rank.append(rank)
                mean_reciprocal_rank.append(1.0 / rank)
                for k in k_values:
                    hits[k].append(1 if rank <= k else 0)
    
    # 计算最终结果
    results = {
        'MR': sum(mean_rank) / len(mean_rank),
        'MRR': sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
    }
    
    for k in k_values:
        results[f'Hits@{k}'] = sum(hits[k]) / len(hits[k])
    
    return results

def filtered_ranking(model, dataset, test_triples=None):
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
    
    results = {
        'Filtered_MR': sum(filtered_mean_rank) / len(filtered_mean_rank),
        'Filtered_MRR': sum(filtered_mean_reciprocal_rank) / len(filtered_mean_reciprocal_rank)
    }
    
    for k in [1, 3, 10]:
        results[f'Filtered_Hits@{k}'] = sum(filtered_hits[k]) / len(filtered_hits[k])
    
    return results

# 使用示例
def evaluate_model(model, dataset):
    # 原始评估
    raw_metrics = evaluate_transe(model, dataset)
    print("\nRaw Metrics:")
    for metric, value in raw_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 过滤后的评估
    filtered_metrics = filtered_ranking(model, dataset)
    print("\nFiltered Metrics:")
    for metric, value in filtered_metrics.items():
        print(f"{metric}: {value:.4f}")

# 在训练后评估模型
evaluate_model(model, dataset)


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_transe_triples(model, dataset, num_samples=100):
    """
    可视化 TransE 三元组关系: h + r ≈ t
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
        head_plus_rel = []  # h + r 的结果
        
        for h, r, t in sampled_triples:
            # 获取embeddings
            h_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[h]])).numpy()
            r_emb = model.relation_embedding(torch.LongTensor([dataset.relation2id[r]])).numpy()
            t_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[t]])).numpy()
            
            heads.append(h_emb[0])
            relations.append(r_emb[0])
            tails.append(t_emb[0])
            head_plus_rel.append(h_emb[0] + r_emb[0])
        
        # 将所有向量堆叠在一起进行降维
        all_vectors = np.vstack([
            heads,
            relations,
            tails,
            head_plus_rel
        ])
        
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(all_vectors)
        
        # 分离降维后的结果
        heads_2d = vectors_2d[:num_samples]
        relations_2d = vectors_2d[num_samples:2*num_samples]
        tails_2d = vectors_2d[2*num_samples:3*num_samples]
        head_plus_rel_2d = vectors_2d[3*num_samples:]
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        # 绘制实体和关系点
        plt.scatter(heads_2d[:, 0], heads_2d[:, 1], c='blue', label='Head', alpha=0.6)
        plt.scatter(relations_2d[:, 0], relations_2d[:, 1], c='green', label='Relation', alpha=0.6)
        plt.scatter(tails_2d[:, 0], tails_2d[:, 1], c='red', label='Tail', alpha=0.6)
        plt.scatter(head_plus_rel_2d[:, 0], head_plus_rel_2d[:, 1], c='purple', label='Head+Relation', alpha=0.6)
        
        # 为每个三元组绘制连接线
        for i in range(num_samples):
            # 头实体到关系的线
            plt.plot([heads_2d[i, 0], relations_2d[i, 0]], 
                    [heads_2d[i, 1], relations_2d[i, 1]], 
                    'gray', alpha=0.2)
            
            # h+r 到尾实体的线
            plt.plot([head_plus_rel_2d[i, 0], tails_2d[i, 0]], 
                    [head_plus_rel_2d[i, 1], tails_2d[i, 1]], 
                    'orange', alpha=0.5)
        
        plt.legend()

        # 保存图像到指定路径
        save_path = "TransE_temporal.png"  # 替换为你的保存路径
        plt.savefig(save_path, dpi=400, bbox_inches='tight')  # 保存图像，设置DPI和边界
        plt.show()
        
        # 计算平均距离误差
        avg_distance = np.mean([
            np.linalg.norm(head_plus_rel_2d[i] - tails_2d[i]) 
            for i in range(num_samples)
        ])
        print(f"Average distance between (h+r) and t in 2D space: {avg_distance:.4f}")

def visualize_specific_relation(model, dataset, relation, num_samples=10):
    """
    可视化特定关系的三元组
    """
    model.eval()
    with torch.no_grad():
        # 收集包含指定关系的三元组
        relation_triples = [
            triple for triple in dataset.triples 
            if triple[1] == relation
        ]
        
        if len(relation_triples) < num_samples:
            num_samples = len(relation_triples)
        
        sampled_triples = np.random.choice(relation_triples, num_samples, replace=False)
        
        # 收集向量
        heads = []
        tails = []
        head_plus_rel = []
        
        r_id = dataset.relation2id[relation]
        r_emb = model.relation_embedding(torch.LongTensor([r_id])).numpy()[0]
        
        for h, _, t in sampled_triples:
            h_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[h]])).numpy()[0]
            t_emb = model.entity_embedding(torch.LongTensor([dataset.entity2id[t]])).numpy()[0]
            
            heads.append(h_emb)
            tails.append(t_emb)
            head_plus_rel.append(h_emb + r_emb)
        
        # t-SNE降维
        all_vectors = np.vstack([heads, tails, head_plus_rel])
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(all_vectors)
        
        heads_2d = vectors_2d[:num_samples]
        tails_2d = vectors_2d[num_samples:2*num_samples]
        head_plus_rel_2d = vectors_2d[2*num_samples:]
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.scatter(heads_2d[:, 0], heads_2d[:, 1], c='blue', label='Head')
        plt.scatter(tails_2d[:, 0], tails_2d[:, 1], c='red', label='Tail')
        plt.scatter(head_plus_rel_2d[:, 0], head_plus_rel_2d[:, 1], c='purple', label='Head+Relation')
        
        # 绘制连接线
        for i in range(num_samples):
            plt.plot([heads_2d[i, 0], head_plus_rel_2d[i, 0]], 
                    [heads_2d[i, 1], head_plus_rel_2d[i, 1]], 
                    'gray', alpha=0.3)
            plt.plot([head_plus_rel_2d[i, 0], tails_2d[i, 0]], 
                    [head_plus_rel_2d[i, 1], tails_2d[i, 1]], 
                    'gray', alpha=0.3)
        
        plt.title(f'TransE Visualization for Relation: {relation}')
        plt.legend()
        plt.show()

# 使用示例
# 可视化随机采样的三元组
visualize_transe_triples(model, dataset, num_samples=53)
