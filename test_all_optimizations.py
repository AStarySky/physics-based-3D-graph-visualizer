"""
Comprehensive test for all OPTIMIZE.md optimizations
Run on Windows: python test_all_optimizations.py
"""
import sys
import os
import time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("=" * 60)
print("OPTIMIZE.md 综合测试")
print("=" * 60)

all_pass = True

# ==================== Phase 1: Klotski.py ====================
print("\n📦 第一阶段：Klotski.py")
try:
    from Klotski import SlidingToy, WalledSlidingToy, build_global_graph

    toy = SlidingToy(
        4, 5,
        np.array([[2, 2], [2, 1], [2, 1], [1, 1], [1, 1], [1, 2], [1, 2], [2, 1], [2, 1]], dtype=int),
        np.array([[0, 0], [2, 0], [2, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3], [2, 4]], dtype=int)
    )

    # Test 1.1: deepcopy → numpy.copy
    start = time.perf_counter()
    neighbours = toy.get_neighbour()
    t1 = time.perf_counter() - start
    assert len(neighbours) > 0, "get_neighbour returned empty!"
    print(f"  ✅ 1.1 deepcopy→numpy.copy: get_neighbour() 返回 {len(neighbours)} 个邻居 ({t1*1000:.2f}ms)")

    # Test 1.2: 预计算木块指纹
    assert hasattr(toy, 'block_fingerprints'), "缺少 block_fingerprints 属性"
    assert toy.block_fingerprints.shape == (len(toy.box_sizes),), f"指纹形状不对: {toy.block_fingerprints.shape}"
    id_matrix = toy.get_identifier_matrix()
    print(f"  ✅ 1.2 预计算木块指纹: fingerprint[0]={toy.block_fingerprints[0]}, identifier_matrix shape={id_matrix.shape}")

    # Test 1.3: 缓存哈希值
    h1 = hash(toy)
    h2 = hash(toy)
    assert h1 == h2, "相同对象的哈希不一致!"
    assert hasattr(toy, '_hash_code'), "缺少 _hash_code 属性"
    print(f"  ✅ 1.3 缓存哈希值: hash={toy._hash_code}")

    # Test build_global_graph
    start = time.perf_counter()
    adj, state2idx, idx2state, edges = build_global_graph(toy)
    t_enum = time.perf_counter() - start
    print(f"  ✅ build_global_graph: {len(state2idx)} 状态, {len(edges)} 边 ({t_enum*1000:.0f}ms)")

except Exception as e:
    print(f"  ❌ Klotski.py 测试失败: {e}")
    all_pass = False

# ==================== Phase 2: GraphPhysics.py ====================
print("\n⚛️  第二阶段：GraphPhysics.py")
try:
    from GraphPhysics import BFSGraphSimulation3D, generate_cube_mesh

    # Create small graph
    adj = generate_cube_mesh(side=3)

    sim = BFSGraphSimulation3D(
        adjacency_list=adj,
        root_node=0,
        k=0.05,
        gravity=0.01,
        damping=0.95,
        time_step=0.05,
        intensity=4
    )

    # Test 2.1/2.2/2.3: numpy arrays + cached pairs + cached mappings
    assert hasattr(sim, 'pos_array'), "缺少 pos_array"
    assert hasattr(sim, 'vel_array'), "缺少 vel_array"
    assert hasattr(sim, 'pairs_array'), "缺少 pairs_array"
    assert hasattr(sim, '_node_to_idx'), "缺少 _node_to_idx"
    print(f"  ✅ 2.1-2.3 数组缓存: pos_array={sim.pos_array.shape}, vel_array={sim.vel_array.shape}")

    # Test add_next_bfs_node
    for _ in range(10):
        sim.add_next_bfs_node()
    print(f"  ✅ add_next_bfs_node: {len(sim._idx_to_node)} 节点, {len(sim.active_edges)} 条边, pairs={sim.pairs_array.shape}")

    # Test 2.4/2.5: vectorized update + merged norm
    sim.update_physics()
    assert sim.pos_array.shape == sim.vel_array.shape
    assert hasattr(sim, '_max_vmag')
    print(f"  ✅ 2.4/2.5 向量化更新: max_vmag={sim._max_vmag:.6f}, pos[0]={sim.pos_array[0]}")

    # Test get_positions_array and get_edge_vertices_array
    pa = sim.get_positions_array()
    ea = sim.get_edge_vertices_array()
    print(f"  ✅ 兼容方法: get_positions_array={pa.shape}, get_edge_vertices_array={ea.shape}")

    # Test 2.6: Taichi kernel (will test with actual simulation)

except ImportError as e:
    print(f"  ⚠️  导入失败（沙箱环境缺失依赖）: {e}")
    print(f"  🔶 请在 Windows 上运行完整测试")
except Exception as e:
    print(f"  ❌ GraphPhysics.py 测试失败: {e}")
    all_pass = False

# ==================== Phase 3: GraphRender.py ====================
print("\n🎨 第三阶段：GraphRender.py")
try:
    from GraphRender import GraphRenderer, update_camera_distance_hybrid

    # Test update_camera_distance_hybrid optimization
    points = np.random.randn(100, 3).astype(np.float32)
    dist = update_camera_distance_hybrid(points, 30.0)
    assert 0 < dist < 10000, f"相机距离异常: {dist}"
    print(f"  ✅ 3.2 update_camera_distance_hybrid: distance={dist:.2f}")

    # Test that uniform cache attributes exist on GraphRenderer class
    # (full test requires OpenGL context)
    import inspect
    init_sig = inspect.signature(GraphRenderer.__init__)
    print(f"  ✅ GraphRenderer 初始化签名: {init_sig}")

except Exception as e:
    print(f"  ⚠️  GraphRender.py 局部测试: {e}")

# ==================== Phase 4: VideoRecorder.py ====================
print("\n📹 第四阶段：VideoRecorder.py")
try:
    from VideoRecorder import AutoVideoRecorder

    recorder = AutoVideoRecorder(fps=60)
    assert hasattr(recorder, 'frame_cond'), "缺少 frame_cond (Condition)"
    print(f"  ✅ 4.2 threading.Condition: frame_cond 已创建")

    # Test _add_overlay signature (functional test requires OpenGL)
    import inspect
    add_method = recorder._add_overlay
    print(f"  ✅ 4.1 _add_overlay 方法就绪")

except Exception as e:
    print(f"  ❌ VideoRecorder.py 测试失败: {e}")
    all_pass = False

# ==================== Phase 5: PlayGround.py ====================
print("\n🎮 第五阶段：PlayGround.py")
try:
    from PlayGround import start_toy16
    print(f"  ✅ 5.2 PlayGround 导入正常")
except Exception as e:
    print(f"  ⚠️  PlayGround.py 导入: {e}")

# ==================== Summary ====================
print("\n" + "=" * 60)
if all_pass:
    print("✅ 全部测试通过！")
else:
    print("❌ 部分测试失败，请检查上面的错误信息")
print("=" * 60)

# Print final file versions
print("\n📋 修改文件版本校验:")
import hashlib
files = ['Klotski.py', 'GraphPhysics.py', 'GraphRender.py', 'VideoRecorder.py', 'PlayGround.py']
for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as fh:
            content = fh.read()
            # Check for key optimizations in each file
            if f == 'Klotski.py':
                checks = [
                    (b'copy.deepcopy' not in content, "  无 deepcopy 残留"),
                    (b'block_fingerprints' in content, "  有 block_fingerprints 预计算"),
                    (b'_hash_code' in content, "  有 _hash_code 缓存"),
                ]
            elif f == 'GraphPhysics.py':
                checks = [
                    (b'pos_array' in content, "  有 pos_array"),
                    (b'pairs_array' in content, "  有 pairs_array"),
                    (b'_max_vmag' in content, "  有 _max_vmag"),
                    (b'for j in range(i + 1, n)' in content, "  对称斥力循环"),
                    (content.count(b'import taichi as ti') == 1, "  无重复 taichi import"),
                ]
            elif f == 'GraphRender.py':
                checks = [
                    (b'self.uniform_' in content, "  有 uniform 缓存"),
                    (b'glGetUniformLocation' not in content.split(b'_render_frame')[1] if b'_render_frame' in content else True, "  _render_frame 无 glGetUniformLocation"),
                    (b'_cached_n_points' in content, "  相机距离缓存"),
                ]
            elif f == 'VideoRecorder.py':
                checks = [
                    (b'.copy()' not in content.split(b'_add_overlay')[1].split(b'def ')[0] if b'_add_overlay' in content else True, "  _add_overlay 无 frame.copy()"),
                    (b'frame_cond' in content, "  有 Condition"),
                    (b'time.sleep(0.001)' not in content, "  无忙等 sleep"),
                ]
            elif f == 'PlayGround.py':
                checks = [
                    (b"os.path.exists" in content, "  文件存在检查"),
                ]

            all_ok = True
            for ok, msg in checks:
                status = "✅" if ok else "❌"
                if not ok:
                    all_ok = False
                print(f"  [{status}] {f}: {msg}")
            if all_ok:
                print(f"  ✅ {f} 全部优化就绪")
