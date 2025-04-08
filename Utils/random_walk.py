import dgl
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_random_walks(g: dgl.DGLGraph, num_walks: int, walk_length: int) -> list:
    walks = []
    nodes = torch.arange(g.number_of_nodes())  # 获取所有节点的索引
    for node in nodes:
        for _ in range(num_walks):  # 每个节点进行 num_walks 次游走
            walk = [node.item()]
            for _ in range(walk_length - 1):
                current_node = walk[-1]
                # 获取当前节点的邻居节点索引
                neighbors = g.successors(current_node).tolist()
                if len(neighbors) == 0:  # 如果没有邻居，则停止游走
                    break
                next_node = random.choice(neighbors)  # 随机选择一个邻居节点
                walk.append(next_node)
            walks.append(walk)
    return walks

def random_walk_encoding(g: dgl.DGLGraph, num_walks: int = 10, walk_length: int = 5) -> np.ndarray:
    """
    基于随机游走生成位置编码。
    :param adj_matrix: 稀疏邻接矩阵 (CSR 格式)
    :param num_walks: 每个节点进行的游走次数
    :param walk_length: 每条游走路径的长度
    :return: 随机游走编码矩阵
    """
    walks = generate_random_walks(g, num_walks, walk_length)

    # 统计每个节点的游走路径，生成编码
    node_walks = {}
    for walk in walks:
        for node in walk:
            if node not in node_walks:
                node_walks[node] = []
            node_walks[node].append(walk)

    # 创建每个节点的随机游走编码
    node_embeddings = {}
    for node, walk_list in node_walks.items():
        # 计算每个节点的编码，这里使用游走路径的均值
        encoded_walk = np.mean([np.array(walk) for walk in walk_list], axis=0)
        node_embeddings[node] = encoded_walk

    # 将编码结果转化为矩阵
    node_indices = sorted(node_embeddings.keys())
    embeddings_matrix = np.array([node_embeddings[node] for node in node_indices])

    return embeddings_matrix

def random_walk_coding(g: dgl):
    random_walk_position_encodding = []
    for i, g in enumerate(dgl.unbatch(g)):
        # 使用原始节点索引
        position_encoding = random_walk_encoding(g, num_walks=10, walk_length=5)
        random_walk_position_encodding.append(position_encoding)
    random_walk_position_encodding = torch.from_numpy(np.vstack(random_walk_position_encodding))
    return random_walk_position_encodding.unsqueeze(0).to(device)

