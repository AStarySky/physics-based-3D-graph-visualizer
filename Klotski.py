import numpy as np
import scipy.sparse as sp
import copy
import networkx as nx
from random import shuffle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

UP = np.array([0, 1])
DOWN = np.array([0, -1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
BUFF = 0.1

class SlidingToy:
    def __init__(self, width, height, box_sizes, box_positions):
        self.width = width
        self.height = height
        self.box_sizes = box_sizes
        self.box_positions = box_positions
        self.covering_matrix = self.get_covering_matrix(box_positions, box_positions + box_sizes)
    
    def __eq__(self, a):
        return (self.get_identifier_matrix() == a.get_identifier_matrix()).all()
    
    def __hash__(self):
        return hash(encode_state(self))
    
    def get_covering_matrix(self, box_dl, box_ur):
        has_block = np.zeros((self.height, self.width))
        for dl, ur in zip(box_dl, box_ur):
            has_block[dl[1]:ur[1], dl[0]:ur[0]] += 1
        return has_block
    
    def isoverlap(self, box_sizes, box_positions):
        box_dl = box_positions
        box_ur = box_positions + box_sizes 
        if (box_dl < 0).any() or (box_ur[:,0] > self.width).any() or (box_ur[:, 1] > self.height).any():
            return True
        else:
            return (self.get_covering_matrix(box_dl, box_ur) > 1).any()
    
    def get_neighbour(self):
        box_dl = self.box_positions
        box_ur = self.box_positions + self.box_sizes
        neigbour_shift = []
        for q, (dl, ur) in enumerate(zip(box_dl, box_ur)):
            if dl[1] > 0:
                if not (self.covering_matrix[dl[1]-1, dl[0]:ur[0]]).any():
                    neigbour_shift.append((q, DOWN))
            if ur[1] < self.height:
                if not (self.covering_matrix[ur[1], dl[0]:ur[0]]).any():
                    neigbour_shift.append((q, UP))
            if dl[0] > 0:
                if not (self.covering_matrix[dl[1]:ur[1], dl[0]-1]).any():
                    neigbour_shift.append((q, LEFT))
            if ur[0] < self.width:
                if not (self.covering_matrix[dl[1]:ur[1], ur[0]]).any():
                    neigbour_shift.append((q, RIGHT))
        neighbours = []
        for shift in neigbour_shift:
            starter = copy.deepcopy(self.box_positions)
            starter[shift[0]] += shift[1]
            neighbours.append(
                SlidingToy(self.width, self.height, self.box_sizes, starter)
            )
        return neighbours

    def get_image_matrix(self):
        has_block = np.zeros((self.height, self.width))
        box_dl = self.box_positions
        box_ur = self.box_positions + self.box_sizes
        for q, (dl, ur) in enumerate(zip(box_dl, box_ur)):
            has_block[dl[1]:ur[1], dl[0]:ur[0]] += q + 1
        return has_block

    def get_identifier_matrix(self):
        has_block = np.zeros((self.height, self.width))
        box_dl = self.box_positions
        box_ur = self.box_positions + self.box_sizes
        for q, (dl, ur) in enumerate(zip(box_dl, box_ur)):
            has_block[dl[1]:ur[1], dl[0]:ur[0]] += np.prod(np.array([2, 3]) ** self.box_sizes[q])
        return has_block
    
    def render_illustrate_figure(self, filename='Figure.png'):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#0B0B49")
        ax.set_facecolor("#0B0B49")
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(5)
        patches = [Rectangle(xy=(coord[0]+BUFF, coord[1]+BUFF), width=width-2*BUFF, height=height-2*BUFF) for coord, (width, height) in zip(self.box_positions, self.box_sizes)]
        shuffle(patches)
        for i, patch in enumerate(patches):
            patch.set(color=mpl.colormaps["hsv"](i / len(patches)))
            ax.add_artist(patch)
        ax.set_xlim(-BUFF, self.width + BUFF)
        ax.set_ylim(-BUFF, self.height + BUFF)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
        
        
def encode_state(slidingtoy):
    """把SlidingToy的状态转为tuple（可hash）"""
    return tuple(map(tuple, slidingtoy.get_identifier_matrix()))

def build_global_graph(start_toy, maxiter = np.inf):
    """
    从初始SlidingToy状态出发，枚举整个有限状态空间，返回无向图的稀疏邻接矩阵
    """
    state2idx = {}
    idx2state = []
    edges = []

    # 初始层
    start_code = encode_state(start_toy)
    state2idx[start_code] = 0
    idx2state.append(start_toy)
    current_visitors = [start_toy]

    counter = 0
    while current_visitors != [] and counter < maxiter:
        next_visitors = []
        counter += 1

        for current in current_visitors:
            cur_idx = state2idx[encode_state(current)]
            for neigh in current.get_neighbour():
                code = encode_state(neigh)
                if code not in state2idx:
                    state2idx[code] = len(state2idx)
                    idx2state.append(neigh)
                    next_visitors.append(neigh)
                neigh_idx = state2idx[code]
                # 无向边
                if cur_idx < neigh_idx:
                    edges.append((cur_idx, neigh_idx))

        current_visitors = next_visitors  # 下一层
        print(len(state2idx))

    # 构造邻接矩阵（无向）
    n = len(state2idx)
    if edges:
        row, col = zip(*edges)
        data = np.ones(len(row), dtype=np.int8)
    else:
        row, col, data = [], [], []
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n))

    return adj.tocsr(), state2idx, idx2state, edges

class WalledSlidingToy(SlidingToy):
    def get_neighbour(self):
        box_dl = self.box_positions
        box_ur = self.box_positions + self.box_sizes
        neigbour_shift = []
        for q, (dl, ur) in enumerate(zip(box_dl, box_ur)):
            if self.box_sizes[q, 0] == 1:
                if dl[1] > 0:
                    if not self.covering_matrix[dl[1]-1, dl[0]]:
                        neigbour_shift.append((q, DOWN))
                if ur[1] < self.height :
                    if not self.covering_matrix[ur[1], dl[0]]:
                        neigbour_shift.append((q, UP))
            if self.box_sizes[q, 1] == 1:
                if dl[0] > 0:
                    if not self.covering_matrix[dl[1], dl[0]-1]:
                        neigbour_shift.append((q, LEFT))
                if ur[0] < self.width:
                    if not self.covering_matrix[dl[1], ur[0]]:
                        neigbour_shift.append((q, RIGHT))
        neighbours = []
        for shift in neigbour_shift:
            starter = copy.deepcopy(self.box_positions)
            starter[shift[0]] += shift[1]
            neighbours.append(
                WalledSlidingToy(self.width, self.height, self.box_sizes, starter)
            )
        return neighbours
    
    def render_illustrate_figure(self, filename='Figure.png'):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#0B0B49")
        ax.set_facecolor("#0B0B49")
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(5)
        patches = [Rectangle(xy=(coord[0]+BUFF, coord[1]+BUFF), width=width-2*BUFF, height=height-2*BUFF) for coord, (width, height) in zip(self.box_positions, self.box_sizes)]
        shuffle(patches)
        for i, patch in enumerate(patches):
            patch.set(color=mpl.colormaps["hsv"](i / len(patches)))
            ax.add_artist(patch)
        ax.set_xlim(-BUFF, self.width + BUFF)
        ax.set_ylim(-BUFF, self.height + BUFF)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=150)

def plot_graph_and_states(adj, idx2state, N=90):
    """
    在一个figure里画两部分：
    左边：全局邻接矩阵
    右边：前N个状态的covering_matrix，网格排列
    """
    num_states = len(idx2state)
    N = min(N, num_states)

    # figure划分成左右两部分，左边1个subplot，右边再细分
    fig = plt.figure(figsize=(12, 6))

    # 左边：邻接矩阵
    ax_left = fig.add_subplot(1, 2, 1)
    ax_left.imshow(adj.toarray(), cmap="Greys", interpolation="none")
    ax_left.set_title("Adjacency Matrix")
    ax_left.set_xlabel("State Index")
    ax_left.set_ylabel("State Index")

    # 右边：covering_matrix 网格
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))

    # 使用 subplot2grid，在右半边生成一个小网格
    for i in range(N):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(rows, cols, i+1, 
                             position=[
                                 0.52 + 0.4/cols*c, 
                                 0.1 + 0.8/rows*(rows-1-r), 
                                 0.3/cols, 0.7/rows])
        toy = idx2state[i]
        ax.imshow(toy.get_image_matrix(), cmap="Blues", origin="lower")
        ax.set_title(f"{i}", fontsize=8)
        ax.axis("off")

    # plt.tight_layout()
    plt.show()

def draw_state_graph_3d(adj, idx2state, N=None):
    """
    adj: 稀疏邻接矩阵 (scipy.sparse)
    idx2state: 状态列表
    N: 只画前N个状态（可选）
    """
    # 转成networkx图
    G = nx.Graph(adj)

    if N is not None:
        nodes = list(range(min(N, len(idx2state))))
        G = G.subgraph(nodes)

    # 3D spring layout
    pos = nx.spring_layout(G, dim=3, seed=41)

    # 提取节点坐标
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]
    zs = [pos[i][2] for i in G.nodes()]

    # 画3D图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 画边
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color="black", linewidth=0.5, alpha=0.6)

    # 画节点
    ax.scatter(xs, ys, zs, size=60, c=np.arange(len(xs)), edgecolors="k", depthshade=True, cmap="viridis")

    # 可以加标签
    # for i, (x, y, z) in pos.items():
    #     ax.text(x, y, z, str(i), fontsize=8)

    ax.set_title("SlidingToy State Graph (3D spring layout)")
    plt.show()

def draw_state_graph_2d(adj, idx2state, N=None):
    """
    adj: 稀疏邻接矩阵 (scipy.sparse)
    idx2state: 状态列表
    N: 只画前N个状态（可选）
    """
    # 转成networkx图
    G = nx.Graph(adj)

    if N is not None:
        nodes = list(range(min(N, len(idx2state))))
        G = G.subgraph(nodes)

    # 3D spring layout
    pos = nx.spring_layout(G)

    # 提取节点坐标
    xs = [pos[i][0] for i in G.nodes()]
    ys = [pos[i][1] for i in G.nodes()]

    # 画3D图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()

    # 画边
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="black", linewidth=0.5, alpha=0.6)

    # 画节点
    # ax.scatter(xs, ys, s=60, c="skyblue", edgecolors="k", depthshade=True)

    # 可以加标签
    # for i, (x, y, z) in pos.items():
    #     ax.text(x, y, z, str(i), fontsize=8)

    ax.set_title("SlidingToy State Graph (2D spring layout)")
    plt.show()


def force_directed_layout(adj_matrix, edges, dim=3, iterations=200, k_s=1, k_r=0.1, L=None, dt=0.01):
    n = adj_matrix.shape[0]
    
    G = nx.Graph(adj_matrix)
    if L is None:
        L = 1/np.sqrt(n)
    # 3D spring layout
    positions = nx.spring_layout(G, dim=3, seed=232, k=L, scale=None)

    # 提取节点坐标
    # pos = np.random.randn(n, dim)  # 随机初始化位置
    pos = np.array([positions[i] for i in G.nodes()])
    # vel = np.zeros((n, dim))       # 初始速度为0
    
    # for _ in range(iterations):
    #     forces = np.zeros((n, dim))
        
    #     # 斥力 (库伦排斥)
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             delta = pos[i] - pos[j]
    #             dist = np.linalg.norm(delta) + 1e-6
    #             force = k_r * delta / dist**3  # ~ 1/d^2
    #             forces[i] += force
    #             forces[j] -= force
        
    #     # 吸引力 (弹簧)
    #     for i, j in edges:
    #         if i > j:
    #             delta = pos[j] - pos[i]
    #             dist = np.linalg.norm(delta) + 1e-6
    #             force = k_s * (dist - L) * delta / dist
    #             forces[i] += force                
    #             forces[j] -= force
        
    #     # 更新速度和位置（阻尼防止震荡）
    #     vel = 0.9 * vel + dt * forces
    #     pos += vel
    
    # 按照协方差矩阵的主轴旋转
    pos -= pos.mean(axis=0)
    cov = np.cov(pos, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    pos = pos @ eigvecs
    
    return pos