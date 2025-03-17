import sys
sys.path.append('../')
from KG import spatial_triples, temporal_triples  #空间三元组和时间三元组
from model import train_transe, train_transh
from evaluate import evaluate_transhmodel, evaluate_tranemodel

# 空间三元组训练TransE和TransH模型
transe_model, transe_dataset = train_transe(spatial_triples, epochs=500)
transh_model, transh_dataset = train_transh(spatial_triples, epochs=500)
# 在训练后评估模型

print("空间三元组评估TransE模型结果")
evaluate_tranemodel(transe_model, transe_dataset)
print("空间三元组评估TransH模型结果")
evaluate_transhmodel(transh_model, transh_dataset)

# 时间三元组训练TransE和TransH模型
transe_model, transe_dataset = train_transe(temporal_triples, epochs=500)
transh_model, transh_dataset = train_transh(temporal_triples, epochs=500)
# 在训练后评估模型
print("时间三元组评估TransE模型结果")
evaluate_tranemodel(transe_model, transe_dataset)
print("时间三元组评估TransH模型结果")
evaluate_transhmodel(transh_model, transh_dataset)