import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher

# 连接到 Neo4j 数据库
test_graph = Graph(
    "bolt://localhost:7687",  # 确保使用正确的协议和端口号
    auth=("neo4j", "20040614")  # 使用 auth 参数传递用户名和密码
)

# 清空现有数据
test_graph.delete_all()

# 读取 Excel 文件中的空间三元组
spatial_triples_df = pd.read_excel("SKG_triples.xlsx")

# 定义需要单独设置为 ocean 标签的实体列表
ocean_entities = ["SST", "Chl-a", "PIC", "POC", "PAR", "NFLH"]

# 定义一个函数来创建节点和关系
def create_triples_in_neo4j(df, graph):
    node_matcher = NodeMatcher(graph)
    for _, row in df.iterrows():
        head = row["头实体"]
        relation = row["关系"]
        tail = row["尾实体"]

        # 检查头实体和尾实体是否已经存在，如果不存在则创建
        head_node = node_matcher.match(name=head).first()
        if not head_node:
            # 如果头实体在 ocean_entities 列表中，则使用 ocean 标签
            head_label = "ocean" if head in ocean_entities else "Entity"
            head_node = Node(head_label, name=head)
            graph.create(head_node)

        tail_node = node_matcher.match(name=tail).first()
        if not tail_node:
            # 如果尾实体在 ocean_entities 列表中，则使用 ocean 标签
            tail_label = "ocean" if tail in ocean_entities else "Entity"
            tail_node = Node(tail_label, name=tail)
            graph.create(tail_node)

        # 创建关系
        rel = Relationship(head_node, relation, tail_node)
        graph.create(rel)

# 将空间三元组存储到 Neo4j 中
create_triples_in_neo4j(spatial_triples_df, test_graph)

# 确认数据已成功导入
result = test_graph.run("MATCH (n)-[r]->(m) RETURN n, r, m")
for record in result.data():
    print(record)