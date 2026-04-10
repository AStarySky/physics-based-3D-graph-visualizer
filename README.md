# 3D Graph Physics & State-Space Visualizer

这是一个基于 **Taichi Lang** 加速的 3D 力导向图（Force-directed Graph）物理仿真与渲染系统。该项目能够自动探索复杂游戏（如**华容道 Klotski**）的状态空间，并将其演化过程可视化为具有物理特性的 3D 动态结构。

## 🌟 核心特性

* **GPU 加速仿真**：利用 `taichi` 编写高性能 Kernel，在 GPU 上并行计算节点间的斥力（Repulsion）与引力（Attraction），支持数千节点级别的实时物理模拟。
* **动态 BFS 演化**：支持从根节点开始，通过广度优先搜索（BFS）逐步“生长”图结构，并伴随平滑的物理过渡动画。
* **3D OpenGL 渲染**：基于 `PyOpenGL` 和 `glfw` 构建的高性能渲染引擎，支持实时交互（旋转、缩放）以及相机自动跟随。
* **自动化视频导出**：集成 `OpenCV` 录制功能，可自动捕捉图的生长过程并导出为高质量的 `.mp4` 视频，包含实时状态遮罩（Overlay）。
* **状态空间搜索**：内置华容道（Sliding Toy）求解算法，能自动将游戏的所有合法步数转化为图的节点与边。

## 🛠️ 技术栈

* **物理引擎**: [Taichi Lang](https://www.taichi-lang.org/)
* **图形渲染**: OpenGL (Modern Core Profile 3.3+)
* **数学计算**: NumPy, PyGLM
* **视频处理**: OpenCV (cv2)
* **图算法**: NetworkX, SciPy

## 📂 文件说明

| 文件名 | 描述 |
| :--- | :--- |
| `PlayGround.py` | **主入口文件**。配置参数并启动仿真与视频导出任务。 |
| `GraphPhysics.py` | 物理引擎核心。包含 Taichi Kernel 实现和 3D BFS 模拟逻辑。 |
| `GraphRender.py` | 基于 OpenGL 的渲染器，处理着色器、缓冲区及相机控制。 |
| `Klotski.py` | 华容道游戏逻辑与状态空间生成算法。 |
| `VideoRecorder.py` | 视频录制工具类，支持异步帧写入和信息叠加。 |
| `GraphHelper.py` | 辅助工具，处理矩阵与邻接表之间的转换。 |

## 🚀 快速开始

### 1. 安装依赖
本项目只在 Python 3.13 进行测试，不保证其他版本的稳定性。要安装依赖，可以运行：
```bash
pip install -r requirements.txt
```

### 2. 运行项目
直接运行 `PlayGround.py` 来生成一个基于华容道状态空间的 3D 演化视频：
```bash
python PlayGround.py
```
程序运行完成后，你将在项目根目录下看到生成的 `my_graph.mp4`。

### 3. 实时交互模式
如果你想开启实时窗口查看并手动操作相机，请修改 `PlayGround.py` 中的渲染器配置：
```python
# 将 visible 设为 True，并调用 run_interactive
renderer = GraphRenderer(window_size=(1280, 720), visible=True)
renderer.run_interactive(sim)
```

## 📊 物理模型说明

系统采用了改进的力导向模型：
* **斥力（Repulsion）**：遵循库仑定律（Coulomb's Law）的变体，节点间产生 $F=k^2/d$ 的排斥力。
* **引力（Attraction）**：基于边连接的弹簧模型，节点之间产生 $F=d^2/k$ 的吸引力，使相邻节点保持合理距离。
* **中心重力**：轻微的向心力防止图结构在无限空间中飘散。
* **阻尼（Damping）**：消耗系统动能，使图结构最终趋于稳定。
