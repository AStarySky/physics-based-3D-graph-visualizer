import numpy as np
from typing import Dict, List, Tuple

def matrix_to_adjacency_list(adj_matrix: np.ndarray) -> Dict[int, List[int]]:
    """
    将邻接矩阵转换为邻接表
    
    Args:
        adj_matrix: 形状为 (N, N) 的邻接矩阵，非零值表示边
        
    Returns:
        邻接表字典 {node_id: [neighbor1, neighbor2, ...]}
    """
    adj_list = {}
    n = adj_matrix.shape[0]
    
    for i in range(n):
        # 找出第 i 行的所有非零元素的列索引
        neighbors = np.where(adj_matrix[i] != 0)[0].tolist()
        if neighbors:  # 只添加有邻居的节点
            adj_list[i] = neighbors
    
    return adj_list

def edges_to_adjacency_list(edges: List[Tuple[int, int]], 
                            directed: bool = False) -> Dict[int, List[int]]:
    """
    将边列表转换为邻接表
    
    Args:
        edges: 边列表，每个元素为 (u, v) 表示一条边
        directed: 是否为有向图（默认为无向图）
        
    Returns:
        邻接表字典 {node_id: [neighbor1, neighbor2, ...]}
    """
    adj_list = {}
    
    for u, v in edges:
        # 添加 u -> v
        if u not in adj_list:
            adj_list[u] = []
        adj_list[u].append(v)
        
        # 如果是无向图，添加 v -> u
        if not directed:
            if v not in adj_list:
                adj_list[v] = []
            adj_list[v].append(u)
    
    return adj_list