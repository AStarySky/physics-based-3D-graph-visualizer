from GraphHelper import *
from GraphPhysics import *
from GraphRender import *
from Klotski import *

np.random.seed(0)

start_toy = SlidingToy(
    4, 5,
    np.array([[1, 2], [1, 2], [1, 2], [1, 2], [2, 2], [2, 1], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[0, 0], [0, 2], [3, 0], [3, 2], [1, 0], [1, 2], [0, 4], [3, 4], [1, 3], [2, 3]], dtype=int)
)

start_toy2 = WalledSlidingToy(
    6, 6,
    np.array([[1, 2], [3, 1], [1, 2], [2, 1], [1, 2], [2, 1], [2, 1], [1, 2], [1, 2], [1, 2], [1, 2], [2, 1], [3, 1]], dtype=int),
    np.array([[1, 0], [2, 0], [5, 0], [0, 2], [2, 1], [0, 3], [2, 3], [4, 2], [5, 2], [0, 4], [2, 4], [4, 4], [3, 5]], dtype=int)
)

start_toy3 = SlidingToy(
    4, 4,
    np.array([[1, 2], [2, 1], [2, 2]], dtype=int),
    np.array([[2, 0], [0, 2], [0, 0]], dtype=int)
)

start_toy4 = SlidingToy(
    4, 5,
    np.array([[1, 2], [1, 2], [1, 2], [1, 2], [2, 2], [2, 1], [2, 1]], dtype=int),
    np.array([[0, 0], [0, 2], [3, 0], [3, 2], [1, 0], [1, 2], [1, 3]], dtype=int)
)

start_toy5 = SlidingToy(
    4, 5,
    np.array([[1, 2], [1, 2], [2, 2], [2, 1], [2, 1], [2, 1]], dtype=int),
    np.array([[0, 0], [3, 0], [1, 0], [1, 2], [1, 3], [1, 4]], dtype=int)
)

start_toy6 = SlidingToy(
    4, 6,
    np.array([[1, 2], [1, 2], [1, 2], [1, 3], [2, 2], [2, 1], [1, 2], [1, 2], [1, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[0, 0], [0, 2], [3, 0], [3, 2], [1, 0], [1, 2], [1, 3], [2, 3], [0, 4], [0, 5], [3, 5]], dtype=int)
)

_, _, _, edges = build_global_graph(start_toy6)
# # 生成图
adj_list = edges_to_adjacency_list(edges)

# 创建模拟实例
sim = BFSGraphSimulation3D(
    adjacency_list=adj_list,
    root_node=0,
    k=0.05,
    gravity=0.01,
    damping=0.95,
    time_step=0.05,
    intensity=4
)

renderer = GraphRenderer(window_size=(1920, 1080), visible=True, line_width=1.0, point_size=2.0)  # 离屏
# 导出视频
renderer.render_video(
    sim,
    "Graph6.mp4",
    viewing_duration=10.0,   # 观赏10秒
    nodes_per_frame=100,       # 每帧添加4个节点
    physics_iter_per_frame=6,
    fps=60
)
# renderer.run_interactive(
#     sim
# )

renderer.close()