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

start_toy7 = SlidingToy(
    4, 6,
    np.array([[2, 1], [1, 3], [1, 3], [1, 2], [3, 1], [1, 4], [1, 1]], dtype=int),
    np.array([[1, 0], [3, 0], [1, 1], [3, 3], [1, 5], [2, 1], [1, 4]],dtype=int)
)

start_toy8 = SlidingToy(
    4, 5,
    np.array([[2, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [1, 2], [1, 3], [2, 2], [2, 3]],dtype=int)
)

start_toy9 = SlidingToy(
    4, 5,
    np.array([[2, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [1, 2], [1, 3], [2, 3]],dtype=int)
)


start_toy10 = WalledSlidingToy(
    6, 6,
    np.array([[3, 1], [1, 3], [2, 1], [1, 2], [2, 1], [1, 2], [1, 2], [1, 2], [2, 1], [3, 1]], dtype=int), 
    np.array([[0, 0], [5, 0], [3, 2], [2, 2], [4, 3], [3, 3], [0, 4], [2, 4], [4, 4], [3, 5]], dtype=int)
)

start_toy11 = WalledSlidingToy(
    6, 6,
    np.array([[1, 3], [2, 1], [1, 3], [2, 1], [1, 2], [2, 1]], dtype=int),
    np.array([[1, 0], [3, 0], [5, 0], [3, 2], [3, 3], [4, 4]], dtype=int)
)

start_toy12 = SlidingToy(
    4, 5,
    np.array([[2, 2], [2, 1], [2, 1], [1, 1], [1, 1], [1, 2], [1, 2], [2, 1], [2, 1]], dtype=int),
    np.array([[0, 0], [2, 0], [2, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3], [2, 4]], dtype=int)
)

start_toy13 = SlidingToy(
    5, 5,
    np.array([[2, 1], [2, 1], [2, 2], [2, 2], [1, 3], [2, 1], [3, 1]], dtype=int),
    np.array([[0, 0], [2, 0], [0, 1], [2, 1], [4, 0], [0, 3], [2, 3]], dtype=int)
)

start_toy14 = WalledSlidingToy(
    6, 6,
    np.array([[1, 3], [2, 1], [1, 2], [1, 2], [1, 2], [2, 1], [1, 3], [3, 1], [1, 2], [1, 2], [2, 1], [2, 1], [2, 1]], dtype=int),
    np.array([[0, 0], [1, 0], [1, 1], [2, 1], [4, 0], [3, 2], [5, 1], [0, 3], [3, 3], [2, 4], [4, 4], [0, 5], [3, 5]], dtype=int)
)

start_toy15 = WalledSlidingToy(
    6, 6,
    np.array([[2, 1], [1, 3], [2, 1], [1, 3], [1, 3], [3, 1], [1, 2], [2, 1]], dtype=int),
    np.array([[0, 0], [0, 1], [1, 2], [3, 1], [5, 0], [2, 5], [0, 4], [4, 4]], dtype=int)
)

start_toy16 = WalledSlidingToy(
    6, 6,
    np.array([[3, 1], [1, 2], [1, 2], [2, 1], [1, 3], [1, 3], [2, 1], [2, 1], [1, 2], [1, 2], [2, 1], [2, 1], [2, 1]]),
    np.array([[0, 0], [3, 0], [0, 1], [1, 1], [4, 1], [5, 1], [2, 2], [0, 3], [2, 3], [1, 4], [4, 4], [2, 5], [4, 5]])
)

# toys = [start_toy, start_toy2, start_toy3, start_toy4, start_toy5, start_toy6, start_toy7, start_toy8, start_toy9, start_toy10, start_toy11, start_toy12, start_toy13, start_toy14, start_toy15, start_toy16]
# for i, s in enumerate(toys):
#     s.render_illustrate_figure(f"Figure{i+1}.png")

# plt.imshow(start_toy16.get_image_matrix())
# plt.show()

start_toy16.render_illustrate_figure()
_, _, _, edges = build_global_graph(start_toy16)
# 生成图
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

renderer = GraphRenderer(window_size=(1920, 1080), visible=True, line_width=1.0, point_size=2.5)  # 离屏
# 导出视频
renderer.render_video(
    sim,
    "Graph16.mp4",
    viewing_duration=10.0,   # 观赏10秒
    nodes_per_frame=100,       # 每帧添加4个节点
    physics_iter_per_frame=3,
    fps=60
)
renderer.run_interactive(
    sim
)

renderer.close()