import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import deque
import taichi as ti

# 选择后端：可以自动检测，或强制指定（如 ti.cuda, ti.vulkan, ti.metal, ti.cpu）
ti.init(arch=ti.vulkan, default_fp=ti.f32)

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

        for j in range(i + 1, n):
            diff = p_i - pos[j]
            dist_sq = diff.norm_sqr() + softening_sq

            f = (k_sq / dist_sq) * diff
            f_i += f
            forces[j] -= f

        forces[i] += f_i

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
    def __init__(self, adjacency_list, root_node, k=25.0, gravity=0.02, damping=0.8, time_step=0.15, intensity=1.0):
        self.adj = adjacency_list
        self.root = root_node
        self.k, self.k_sq = k, k * k
        self.gravity = gravity
        self.damping = damping
        self.dt = time_step
        self.intensity = intensity

        self.pos_array = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.vel_array = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self._node_to_idx = {root_node: 0}
        self._idx_to_node = [root_node]

        self.active_edges = set()
        self.pairs_array = np.empty((0, 2), dtype=np.int32)

        self.bfs_queue = deque([root_node])
        self.visited = {root_node}
        self._max_vmag = 0.0
    
    def get_positions_array(self) -> np.ndarray:
        return self.pos_array.astype(np.float32)

    def get_max_vmag(self):
        return self._max_vmag

    def get_edge_vertices_array(self) -> np.ndarray:
        if not self.active_edges:
            return np.empty((0, 3), dtype=np.float32)
        vertices = []
        for u, v in self.active_edges:
            if u in self._node_to_idx and v in self._node_to_idx:
                idx_u = self._node_to_idx[u]
                idx_v = self._node_to_idx[v]
                vertices.append(self.pos_array[idx_u])
                vertices.append(self.pos_array[idx_v])
        if not vertices:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(vertices, dtype=np.float32)

    def add_next_bfs_node(self):
        if not self.bfs_queue:
            return False
        u = self.bfs_queue.popleft()
        if u in self.adj:
            for v in self.adj[u]:
                is_new = v not in self.visited
                if is_new:
                    self.visited.add(v)
                    self.bfs_queue.append(v)
                    idx_u = self._node_to_idx[u]
                    parent_pos = self.pos_array[idx_u]
                    random_dir = np.random.normal(0, 1, 3)
                    random_dir /= np.linalg.norm(random_dir)
                    self.pos_array = np.vstack([self.pos_array, parent_pos + random_dir * self.k])
                    self.vel_array = np.vstack([self.vel_array, np.zeros((1, 3))])
                    self._node_to_idx[v] = len(self._idx_to_node)
                    self._idx_to_node.append(v)

                edge_key = tuple(sorted((u, v)))
                if edge_key not in self.active_edges:
                    self.active_edges.add(edge_key)
                    idx_u = self._node_to_idx[u]
                    idx_v = self._node_to_idx[v]
                    self.pairs_array = np.vstack([self.pairs_array, np.array([[idx_u, idx_v]], dtype=np.int32)])
        return True

    def update_physics(self):
        n = len(self._idx_to_node)
        if n == 0:
            return

        pos_f32 = self.pos_array.astype(np.float32)

        repulsion_forces = np.zeros((n, 3), dtype=np.float32)
        if n > 1 and len(self.pairs_array) > 0:
            compute_repulsion_taichi(pos_f32, repulsion_forces, self.pairs_array, self.k_sq, 1.0 / self.k)

        total_forces = repulsion_forces.astype(np.float64) + self.gravity * (-self.pos_array)

        self.vel_array = (self.vel_array + total_forces * self.dt * self.intensity) * self.damping

        vel_mags = np.linalg.norm(self.vel_array, axis=1)
        over_limit = vel_mags > 10.0
        if over_limit.any():
            self.vel_array[over_limit] = (self.vel_array[over_limit].T / vel_mags[over_limit] * 10.0).T
            vel_mags[over_limit] = 10.0
        self._max_vmag = float(np.max(vel_mags)) if n > 0 else 0.0

        self.pos_array += self.vel_array * self.dt 
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