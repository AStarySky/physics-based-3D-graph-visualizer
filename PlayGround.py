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
_, _, _, edges = build_global_graph(start_toy)
# 生成图

adj_list = edges_to_adjacency_list(edges)

# 创建模拟实例
sim = BFSGraphSimulation3D(
    adjacency_list=adj_list,
    root_node=0,
    k=0.05,
    gravity=0.03,
    damping=0.96,
    time_step=0.05,
    intensity=4
)

renderer = GraphRenderer(window_size=(1920, 1080), visible=False, line_width=0.5, point_size=2.0)  # 离屏
# 导出视频
renderer.render_video(
    sim,
    "my_graph.mp4",
    viewing_duration=10.0,   # 观赏10秒
    nodes_per_frame=4,       # 每帧添加4个节点
    physics_iter_per_frame=6,
    fps=60
)


renderer.close()