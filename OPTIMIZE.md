# 性能优化任务列表

## 第一阶段：状态空间枚举（Klotski.py）

### 1.1 `deepcopy` → `numpy.copy`
- **文件**: Klotski.py
- **位置**: `get_neighbour()` 第 64 行和 WalledSlidingToy 第 181 行
- **现状**: `copy.deepcopy(self.box_positions)` 对纯 numpy 数组调用重量级 deepcopy
- **改进**: 改用 `self.box_positions.copy()`（numpy 原生浅拷贝，对整数数组完全等价且快 10-50 倍）
- **预估收益**: 枚举阶段 BFS 对每个邻居调用一次，总调用次数 = 边数 × 2，中等

### 1.2 预计算木块指纹
- **文件**: Klotski.py
- **位置**: `get_identifier_matrix()` 第 79-85 行
- **现状**: 每次调用时在循环中计算 `np.prod(np.array([2,3]) ** box_sizes[q])`
- **改进**: 在 `__init__` 中预计算 `self.block_fingerprints = 2**box_sizes[:,0] * 3**box_sizes[:,1]`，使用时直接索引
- **预估收益**: 每个状态被哈希/比较时调用 1 次，BFS 中对每个邻居调用，节省重复向量运算

### 1.3 缓存状态哈希值
- **文件**: Klotski.py
- **位置**: `__hash__()` 第 28-29 行
- **现状**: 每次调用 `encode_state(self)` 都重新生成完整标识矩阵再转元组
- **改进**: 在 `__init__` 中计算并缓存 `self._hash_code`，`__hash__` 直接返回
- **预估收益**: 每个状态被 `state2idx` 查找 1-2 次，状态去重的哈希表查找会频繁调 `__hash__`

## 第二阶段：物理仿真引擎（GraphPhysics.py）

### 2.1 节点数据从 dict 迁移为 numpy 数组缓存
- **文件**: GraphPhysics.py
- **位置**: `BFSGraphSimulation3D` 类，`update_physics()` 第 120-156 行
- **现状**: 节点位置/速度存储在 `active_nodes` dict 中，每帧通过列表推导重建 pos numpy 数组
- **改进**: 维护 `self.pos_array` (N,3) 和 `self.vel_array` (N,3) 作为主存储，dict 仅作索引映射。新增节点时通过 `np.vstack` 追加，`update_physics` 直接操作数组
- **预估收益**: **最大**。每帧省去约 3 次完整 Python 列表推导→numpy 数组构建，N=5000 时可省数十毫秒

### 2.2 边对数组缓存
- **文件**: GraphPhysics.py
- **位置**: `update_physics()` 第 128 行
- **现状**: `np.array([[node_to_idx[i], node_to_idx[j]] for i, j in self.active_edges])` 每帧重新构建
- **改进**: 在 `add_next_bfs_node` 中增量维护 `self.pairs_array`，新边加入时 `np.vstack` 追加一行
- **预估收益**: 大。边数可达 2-3 万条，每帧重建边对数组开销显著

### 2.3 节点索引映射缓存
- **文件**: GraphPhysics.py
- **位置**: `update_physics()` 第 127 行
- **现状**: `node_to_idx = {node: i for i, node in enumerate(nodes)}` 每帧重建 dict
- **改进**: 维护 `self._node_to_idx` 和 `self._idx_to_node` 映射，仅在新增节点时更新
- **预估收益**: 与 2.1 配合后自然消除

### 2.4 intial vel 存放位置
- **文件**: GraphPhysics.py
- **位置**: `update_physics()` 第 148-156 行
- **现状**: 总力计算后遍历 Python dict 逐节点更新速度/位置
- **改进**: 配合 2.1 改为向量化操作：`vel = (vel + total_forces * dt * intensity) * damping; pos += vel * dt`
- **预估收益**: 消除 Python for 循环，N=5000 时显著

### 2.5 冗余 vel 范数计算
- **文件**: GraphPhysics.py
- **位置**: `update_physics()` 第 153 行 + `get_max_vmag()` 第 81-85 行
- **现状**: `update_physics` 中对每个节点算一次 `norm`（做速度限幅），`get_max_vmag` 又对所有节点算一次 `norm`
- **改进**: 合并：在 `update_physics` 中一次遍历完成限幅并记录最大值，暴露为属性
- **预估收益**: 小-中。消除一次额外的全节点遍历

### 2.6 Taichi kernel：融合两层循环
- **文件**: GraphPhysics.py
- **位置**: `compute_repulsion_taichi` 第 12-54 行
- **现状**: 斥力循环 O(N²) 中每对都做 `if i == j: continue` 分支
- **改进**: 改为 `for j in range(i+1, n)` + 对称更新，或利用 `ti.loop_config` 优化；考虑 Block 分块进一步利用 shared memory
- **预估收益**: 中。减少约 N 次多余的分支判断，但不改变 O(N²) 本质

## 第三阶段：渲染（GraphRender.py）

### 3.1 缓存 OpenGL uniform 位置
- **文件**: GraphRender.py
- **位置**: `_render_frame()` 第 226-233 行
- **现状**: 每帧 6 次 `glGetUniformLocation` 调用
- **改进**: 在 `_init_shaders` 后查询并缓存到 `self.uniform_*` 属性
- **预估收益**: 小。每次 `glGetUniformLocation` 有 GPU 驱动开销

### 3.2 `_auto_adjust_camera_distance` O(N) 遍历
- **文件**: GraphRender.py
- **位置**: `update_camera_distance_hybrid` 第 20-24 行
- **现状**: 每帧 `np.mean`, `np.max(np.linalg.norm(...))`, `np.max(axis=0)-np.min(axis=0)` 三趟 O(N) 遍历
- **改进**: 用 `scipy.spatial.cKDTree` 或直接在仿真中维护 bbox ± centroid 统计
- **预估收益**: 小。三趟 O(N) 遍历虽然线性但可与仿真合并

## 第四阶段：视频录制（VideoRecorder.py）

### 4.1 减少 overlay 帧拷贝
- **文件**: VideoRecorder.py
- **位置**: `_add_overlay()` 第 204-206 行
- **现状**: `frame.copy()` + `addWeighted` 两次全帧操作
- **改进**: 直接在 `frame` 上画半透明矩形 + putText，省去一次 copy 和 blend
- **预估收益**: 小-中。1920×1080 每帧省去 2MB 拷贝

### 4.2 写入线程用 Condition 替代忙等
- **文件**: VideoRecorder.py
- **位置**: `_writer_loop()` 第 244 行
- **现状**: `time.sleep(0.001)` 忙等式轮询空队列
- **改进**: 使用 `threading.Condition`，生产者 notify，消费者 wait
- **预估收益**: 微小。减少 CPU 空转

## 第五阶段：杂项

### 5.1 PlayGround 重复 taichi import
- **文件**: GraphPhysics.py
- **位置**: 第 6 行和第 10 行
- **现状**: `import taichi as ti` 被导入了两次
- **改进**: 删除第 10 行的重复 import
- **预估收益**: 微小

### 5.2 Graph16.mp4 etc. 已生成视频的清理逻辑
- **文件**: PlayGround.py
- **位置**: 第 131-137 行
- **现状**: 直接覆盖视频，每次都重新渲染
- **改进**: 可加判断：如果视频已存在则跳过，或加 `--force` 标志
- **预估收益**: 易用性提升
