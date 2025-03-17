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

def train_transe(triples, epochs=100, batch_size=32, embedding_dim = 100, lr=0.01):
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
        
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset




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

def train_transh(triples, epochs=100, batch_size=32, embedding_dim=100, lr=0.01, margin=1.0, C=0.1):
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
        
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset

