import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import deque
import taichi as ti

# 选择后端：可以自动检测，或强制指定（如 ti.cuda, ti.vulkan, ti.metal, ti.cpu）
ti.init(arch=ti.vulkan, default_fp=ti.f32)  # 如果无 GPU 则回退到 CPU
import taichi as ti

@ti.kernel
def compute_repulsion_taichi(
    pos: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1),  # (N,) 向量数组
    forces: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1), # (N,) 向量数组
    pairs: ti.types.ndarray(dtype=ti.i32, ndim=2), # (K, 2)
    k_sq: ti.f32,
    inv_k: ti.f32, # 预计算 1.0 / sqrt(k_sq) 传进来
):
    n = pos.shape[0]
    num_pairs = pairs.shape[0]
    softening_sq = 1e-4  # 软化因子的平方

    # 1. 计算斥力 (O(N^2))
    # Taichi 会自动并行化这个最外层循环
    for i in range(n):
        p_i = pos[i]
        f_i = ti.Vector([0.0, 0.0, 0.0])
        
        for j in range(n):
            if i == j: continue
            
            diff = p_i - pos[j]
            dist_sq = diff.norm_sq() + softening_sq
            
            # 斥力公式优化：f = (k^2 / |r|^2) * r_vec
            f_i += (k_sq / dist_sq) * diff
            
        forces[i] = f_i

    # 2. 计算引力 (基于边)
    # 使用 atomic_add 因为多个边可能共享同一个节点
    for p in range(num_pairs):
        idx1 = pairs[p, 0]
        idx2 = pairs[p, 1]
        
        diff = pos[idx1] - pos[idx2]
        dist = diff.norm() + 1e-6
        
        # 引力公式：f = -(|r| / k) * r_vec
        f_grad = -(dist * inv_k) * diff
        
        forces[idx1] += f_grad
        forces[idx2] -= f_grad
            

# --- 1. 3D 物理引擎 ---
class BFSGraphSimulation3D:
    def __init__(self, adjacency_list, root_node, k=25.0, gravity=0.02, damping=0.8, time_step=0.15, intensity = 1.0):
        self.adj = adjacency_list
        self.root = root_node
        self.k, self.k_sq = k, k*k
        self.gravity = gravity
        self.damping = damping
        self.dt = time_step
        self.intensity = intensity
        
        # 3D 坐标与速度
        self.active_nodes = {root_node: {'pos': np.array([0.0, 0.0, 0.0]), 'vel': np.array([0.0, 0.0, 0.0])}}
        self.active_edges = set()
        self.bfs_queue = deque([root_node])
        self.visited = {root_node}
    
    def get_positions_array(self) -> np.ndarray:
        """返回当前所有活动节点位置的 (N,3) numpy 数组，dtype=float32"""
        if not self.active_nodes:
            return np.empty((0, 3), dtype=np.float32)
        # 按节点插入顺序收集位置（顺序不重要，但需保持一致）
        return np.array([data['pos'] for data in self.active_nodes.values()], dtype=np.float32)

    def get_edge_vertices_array(self) -> np.ndarray:
        """返回用于 GL_LINES 的顶点数组：每条边两个端点顺序排列，形状 (M*2, 3)"""
        if not self.active_edges:
            return np.empty((0, 3), dtype=np.float32)
        vertices = []
        for u, v in self.active_edges:
            # 确保边两端节点均存在（理论上一定存在）
            if u in self.active_nodes and v in self.active_nodes:
                vertices.append(self.active_nodes[u]['pos'])
                vertices.append(self.active_nodes[v]['pos'])
        if not vertices:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(vertices, dtype=np.float32)

    def add_next_bfs_node(self):
        if not self.bfs_queue: return False
        u = self.bfs_queue.popleft()
        if u in self.adj:
            for v in self.adj[u]:
                if v not in self.visited:
                    self.visited.add(v)
                    self.bfs_queue.append(v)
                    parent_pos = self.active_nodes[u]['pos']
                    # 在父节点附近产生，给予 3D 随机扰动
                    random_dir = np.random.normal(0, 1, 3)
                    random_dir /= np.linalg.norm(random_dir)
                    self.active_nodes[v] = {
                        'pos': parent_pos + random_dir * self.k, 
                        'vel': np.zeros(3)
                    }
                self.active_edges.add(tuple(sorted((u, v))))
        return True

    def update_physics(self):
        nodes = list(self.active_nodes.keys())
        n = len(nodes)
        if n == 0:
            return

        pos = np.array([self.active_nodes[node]['pos'] for node in nodes], dtype=np.float32)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        pairs = np.array([[node_to_idx[i], node_to_idx[j]] for i, j in self.active_edges], dtype=np.int32)

        # ---------- 斥力计算（Taichi GPU） ----------
        repulsion_forces = np.zeros((n, 3), dtype=np.float32)
        if n > 1:
            compute_repulsion_taichi(pos, self.k_sq, repulsion_forces, pairs)
        # 注意：Taichi 输出的 forces 已是 float32，后续可能需转换为 float64（若需要）

        # ---------- 引力计算（不变） ----------
        # attraction_forces = np.zeros((n, 3))
        # for u, v in self.active_edges:
        #     idx_u, idx_v = node_to_idx[u], node_to_idx[v]
        #     diff = pos[idx_u] - pos[idx_v]
        #     dist = np.linalg.norm(diff) + 0.1
        #     force_mag = (dist**2) / self.k
        #     force_vec = (diff / dist) * force_mag
        #     attraction_forces[idx_u] -= force_vec
        #     attraction_forces[idx_v] += force_vec

        # ---------- 重力与状态更新 ----------
        total_forces = repulsion_forces.astype(np.float64) + self.gravity * (-pos)

        for i, node in enumerate(nodes):
            data = self.active_nodes[node]
            data['vel'] = (data['vel'] + total_forces[i] * self.dt * self.intensity) * self.damping
            velo_mag = np.linalg.norm(data['vel'])
            if velo_mag > 10:
                data['vel'] = data['vel'] / velo_mag * 10
            data['pos'] += data['vel'] * self.dt 
            # if len()
         

# --- 2. 构造 3D 立方体网格数据 ---
def generate_cube_mesh(side=4):
    adj = {}
    def get_id(x, y, z): return x * side**2 + y * side + z
    
    for x in range(side):
        for y in range(side):
            for z in range(side):
                u = get_id(x, y, z)
                neighbors = []
                if x + 1 < side: neighbors.append(get_id(x+1, y, z))
                if y + 1 < side: neighbors.append(get_id(x, y+1, z))
                if z + 1 < side: neighbors.append(get_id(x, y, z+1))
                if x - 1 >= 0: neighbors.append(get_id(x-1, y, z))
                if y - 1 >= 0: neighbors.append(get_id(x, y-1, z))
                if z - 1 >= 0: neighbors.append(get_id(x, y, z-1))
                adj[u] = neighbors
    return adj


# if __name__ == "__main__":
#     from GraphRender import GraphRenderer  # 导入渲染器
#     from Klotski import SlidingToy, WalledSlidingToy, build_global_graph
#     sldtoy = WalledSlidingToy(
#         7, 7, 
#         np.array([[1, 2], [2, 1]]),
#         np.array([[0, 3], [3, 0]])
#     )
#     _, _, _, edges = build_global_graph(sldtoy)
#     # 生成图
#     side = 9
#     adj_list = generate_cube_mesh(side=side)
#     print(f"图规模: {side}^3 = {side**3} 个节点")

#     # 创建模拟实例
#     sim = BFSGraphSimulation3D(
#         adjacency_list=adj_list,
#         root_node=0,
#         k=0.3,
#         gravity=0.01,
#         damping=0.95,
#         time_step=0.05
#     )

#     # 预先添加一些初始节点
#     # for _ in range(50):
#         # sim.add_next_bfs_node()

#     # 创建渲染器
#     renderer = GraphRenderer(window_size=(1280*2, 720*2),
#                              point_color=(0.2, 0.8, 1.0),
#                              edge_color=(0.7, 0.7, 0.7))

#     # 定义每帧更新函数（推进物理和 BFS 展开）
#     def update_simulation():
#         # 每帧更新物理两次，添加两个节点
#         for _ in range(2):
#             sim.update_physics()
#         for _ in range(2):
#             sim.add_next_bfs_node()

#     # 设置回调
#     renderer.set_update_callback(update_simulation)
#     renderer.set_data_callback(lambda: (sim.get_positions_array(), sim.get_edge_vertices_array()))

#     # 启动渲染循环
#     renderer.run()