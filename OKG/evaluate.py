import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
def evaluate_tranemodel(model, dataset):
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




def evaluate_transh(model, dataset, test_triples=None, k_values=[1, 3, 10]):
    """
    评估TransH模型性能
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
            r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
            norm = model.normal_vector(torch.LongTensor([rel_id]))
            t_emb = model.entity_embedding(torch.LongTensor([tail_id]))
            t_proj = model._transfer(t_emb, norm)
            
            for ent_id in range(dataset.num_entities):
                h_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                h_proj = model._transfer(h_emb, norm)
                score = torch.norm(h_proj + r_emb - t_proj, p=2)
                scores_head.append((score.item(), ent_id))
            
            scores_head.sort()
            rank_head = 1
            for score, ent_id in scores_head:
                if ent_id == head_id:
                    break
                rank_head += 1
            
            # Tail prediction
            scores_tail = []
            h_emb = model.entity_embedding(torch.LongTensor([head_id]))
            h_proj = model._transfer(h_emb, norm)
            
            for ent_id in range(dataset.num_entities):
                t_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                t_proj = model._transfer(t_emb, norm)
                score = torch.norm(h_proj + r_emb - t_proj, p=2)
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
    
    results = {
        'MR': sum(mean_rank) / len(mean_rank),
        'MRR': sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
    }
    
    for k in k_values:
        results[f'Hits@{k}'] = sum(hits[k]) / len(hits[k])
    
    return results

def filtered_ranking_transh(model, dataset, test_triples=None):
    """
    计算TransH模型的过滤后排名
    """
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
            r_emb = model.relation_embedding(torch.LongTensor([rel_id]))
            norm = model.normal_vector(torch.LongTensor([rel_id]))
            t_emb = model.entity_embedding(torch.LongTensor([tail_id]))
            t_proj = model._transfer(t_emb, norm)
            
            for ent_id in range(dataset.num_entities):
                if (ent_id, rel_id, tail_id) in all_true_triples and ent_id != head_id:
                    continue
                h_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                h_proj = model._transfer(h_emb, norm)
                score = torch.norm(h_proj + r_emb - t_proj, p=2)
                scores_head.append((score.item(), ent_id))
            
            scores_head.sort()
            filtered_rank_head = 1
            for score, ent_id in scores_head:
                if ent_id == head_id:
                    break
                filtered_rank_head += 1
            
            # Filtered tail prediction
            scores_tail = []
            h_emb = model.entity_embedding(torch.LongTensor([head_id]))
            h_proj = model._transfer(h_emb, norm)
            
            for ent_id in range(dataset.num_entities):
                if (head_id, rel_id, ent_id) in all_true_triples and ent_id != tail_id:
                    continue
                t_emb = model.entity_embedding(torch.LongTensor([ent_id]))
                t_proj = model._transfer(t_emb, norm)
                score = torch.norm(h_proj + r_emb - t_proj, p=2)
                scores_tail.append((score.item(), ent_id))
            
            scores_tail.sort()
            filtered_rank_tail = 1
            for score, ent_id in scores_tail:
                if ent_id == tail_id:
                    break
                filtered_rank_tail += 1
            
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

def evaluate_transhmodel(model, dataset):
    """
    评估TransH模型
    """
    raw_metrics = evaluate_transh(model, dataset)
    print("\nRaw Metrics:")
    for metric, value in raw_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    filtered_metrics = filtered_ranking_transh(model, dataset)
    print("\nFiltered Metrics:")
    for metric, value in filtered_metrics.items():
        print(f"{metric}: {value:.4f}")