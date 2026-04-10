import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glm
import numpy as np
import math
from VideoRecorder import AutoVideoRecorder
import time

def update_camera_distance_hybrid(points: np.ndarray,
                                  current_distance: float,
                                  min_distance: float = 5.0,
                                  max_distance: float = 150.0) -> float:
    """
    混合方法：同时考虑节点数量和空间分布
    """
    if len(points) == 0:
        return current_distance
    
    # 计算质心到最远点的距离
    centroid = np.mean(points, axis=0)
    max_dist = np.max(np.linalg.norm(points - centroid, axis=1))
    
    # 计算包围盒对角线长度
    bbox_size = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    
    # 综合两种度量
    scene_scale = max(max_dist, bbox_size*0.9)
    
    # 根据节点数量微调
    node_factor = np.log(len(points) + 1) * 0.5
    ideal_distance = scene_scale + node_factor
    
    # 平滑过渡
    new_distance = current_distance * 0.92 + ideal_distance * 0.08
    
    return np.clip(new_distance, min_distance, max_distance)

class GraphRenderer:
    """离屏/实时渲染器，支持视频导出"""
    
    def __init__(self, window_size=(1920, 1080), line_width=1.5, point_size=6.0, visible=False):
        self.window_width, self.window_height = window_size
        self.visible = visible
        self.line_width = line_width
        self.point_size = point_size
        self.amb_rot_speed = 60
        self._init_glfw()
        self._init_shaders()
        self._init_buffers()
        
        # 相机参数
        self.camera_target = glm.vec3(0.0)
        self.camera_distance = 30.0
        self.camera_yaw = -90.0
        self.camera_pitch = 0.0
        self.fov = 45.0
        
        # 控制标志
        self.auto_adjust_camera = True
        self.mouse_pressed = False
        self.last_mouse_pos = glm.vec2(0.0)
        
        # 颜色
        self.point_color = glm.vec3(0.2, 0.8, 1.0)
        self.edge_color = glm.vec3(0.7, 0.7, 0.7)
        
        
        # 交互回调（仅当 visible=True 时有效）
        if self.visible:
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
        
        self.last_time = glfw.get_time()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败")
        glfw.window_hint(glfw.VISIBLE, self.visible)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.window_width, self.window_height,
                                         "Graph Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("窗口创建失败")
        glfw.make_context_current(self.window)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(self.point_size)
        glLineWidth(self.line_width)

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = (action == glfw.PRESS)
            if self.mouse_pressed:
                x, y = glfw.get_cursor_pos(window)
                self.last_mouse_pos = glm.vec2(x, y)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not self.mouse_pressed:
            return
        delta = glm.vec2(xpos, ypos) - self.last_mouse_pos
        self.last_mouse_pos = glm.vec2(xpos, ypos)
        self.camera_yaw   += delta.x * 0.3
        self.camera_pitch -= delta.y * 0.3
        self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))

    def _scroll_callback(self, window, xoffset, yoffset):
        self.camera_distance -= yoffset * 1.0
        self.camera_distance = max(1.0, self.camera_distance)

    def _init_shaders(self):
        vertex_src = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
        """
        fragment_src = """
        #version 330 core
        out vec4 FragColor;
        uniform vec3 color;
        void main() {
            FragColor = vec4(color, 1.0);
        }
        """
        vs = compileShader(vertex_src, GL_VERTEX_SHADER)
        fs = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        self.shader = compileProgram(vs, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)

    def _init_buffers(self):
        self.point_vao = glGenVertexArrays(1)
        self.point_vbo = glGenBuffers(1)
        self.edge_vao = glGenVertexArrays(1)
        self.edge_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.point_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        glBindVertexArray(self.edge_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.edge_vbo)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def update_buffers(self, points: np.ndarray, edges: np.ndarray):
        glBindBuffer(GL_ARRAY_BUFFER, self.point_vbo)
        glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_DYNAMIC_DRAW)
        self.point_count = points.shape[0]
        
        glBindBuffer(GL_ARRAY_BUFFER, self.edge_vbo)
        glBufferData(GL_ARRAY_BUFFER, edges.nbytes, edges, GL_DYNAMIC_DRAW)
        self.edge_vertex_count = edges.shape[0]
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def get_view_matrix(self) -> glm.mat4:
        yaw_rad = glm.radians(self.camera_yaw)
        pitch_rad = glm.radians(self.camera_pitch)
        pos = self.camera_target + glm.vec3(
            self.camera_distance * math.cos(yaw_rad) * math.cos(pitch_rad),
            self.camera_distance * math.sin(pitch_rad),
            self.camera_distance * math.sin(yaw_rad) * math.cos(pitch_rad)
        )
        return glm.lookAt(pos, self.camera_target, glm.vec3(0.0, 1.0, 0.0))

    def _auto_adjust_camera_distance(self, points: np.ndarray):
        if len(points) == 0:
            return
        self.camera_distance = update_camera_distance_hybrid(
            points, 
            self.camera_distance,
            min_distance=5.0,
            max_distance=200.0
        )

    def _render_frame(self):
        """执行一次渲染（不交换缓冲区）"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        aspect = self.window_width / self.window_height
        projection = glm.perspective(glm.radians(self.fov), aspect, 0.1, 1000.0)
        view = self.get_view_matrix()
        
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 
                           1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 
                           1, GL_FALSE, glm.value_ptr(view))
        
        if hasattr(self, 'edge_vertex_count') and self.edge_vertex_count > 0:
            glUniform3fv(glGetUniformLocation(self.shader, "color"), 
                         1, glm.value_ptr(self.edge_color))
            glBindVertexArray(self.edge_vao)
            glDrawArrays(GL_LINES, 0, self.edge_vertex_count)
        
        if hasattr(self, 'point_count') and self.point_count > 0:
            glUniform3fv(glGetUniformLocation(self.shader, "color"), 
                         1, glm.value_ptr(self.point_color))
            glBindVertexArray(self.point_vao)
            glDrawArrays(GL_POINTS, 0, self.point_count)
        
        glBindVertexArray(0)
        glFlush()

    def render_video(self, sim, output_path: str,
                     viewing_duration: float = 8.0,
                     nodes_per_frame: int = 2,
                     physics_iter_per_frame: int = 6,
                     fps: int = 60):
        """
        导出视频（自动生长 + 观赏旋转）
        
        Args:
            sim: BFSGraphSimulation3D 实例（必须具有 active_nodes, adj, add_next_bfs_node, update_physics, get_positions_array, get_edge_vertices_array）
            output_path: 输出视频路径
            viewing_duration: 生长完成后的观赏时间（秒）
            nodes_per_frame: 每帧添加的节点数（控制生长速度）
            fps: 视频帧率
        """
        # 获取总节点数
        total_nodes = len(sim.adj) if hasattr(sim, 'adj') else len(sim.visited)
        
        recorder = AutoVideoRecorder(fps=fps)
        recorder.start_auto_render(
            output_path, self.window_width, self.window_height,
            total_nodes=total_nodes,
            viewing_duration=viewing_duration,
            nodes_per_frame=nodes_per_frame
        )
        
        original_auto_adjust = self.auto_adjust_camera
        self.auto_adjust_camera = True
        self._original_camera_state = None
        
        print(f"\n🎬 开始渲染图动画（共 {total_nodes} 个节点）...")
        
        while recorder.is_recording:
            current_time = glfw.get_time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            
            current_nodes = len(sim.active_nodes)
            status = recorder.update(delta_time, current_nodes)
            
            # 生长阶段：添加节点
            if recorder.should_add_nodes():
                for _ in range(recorder.nodes_per_frame):
                    if not sim.add_next_bfs_node():
                        break
            
            # 物理步进
            if recorder.phase.name == 'GROWTH':
                for i in range(physics_iter_per_frame):
                    sim.update_physics()
            
            points = sim.get_positions_array()
            edges = sim.get_edge_vertices_array()
            
            # 自动调整相机（仅生长阶段）
            if recorder.phase.name == 'GROWTH' and self.auto_adjust_camera:
                self._auto_adjust_camera_distance(points)
            
            # 观赏阶段：保存初始相机状态并应用旋转
            # if recorder.phase.name == 'VIEWING':
            #     if self._original_camera_state is None:
            #         self._original_camera_state = {
            #             'yaw': self.camera_yaw,
            #             'pitch': self.camera_pitch,
            #             'distance': self.camera_distance,
            #             'target': glm.vec3(self.camera_target)
            #         }
            self.camera_yaw += self.amb_rot_speed * 1/fps
            
            self.update_buffers(points, edges)
            self._render_frame()
            # if self.visible:
            glfw.swap_buffers(self.window)
            recorder.capture_frame()
            
            # 进度打印
            if recorder.total_frames % 30 == 0:
                if recorder.phase.name == 'GROWTH':
                    print(f"  🌱 生长进度: {current_nodes}/{total_nodes} 节点 ({current_nodes/total_nodes*100:.1f}%)")
                else:
                    elapsed = time.time() - recorder.phase_start_time
                    remaining = viewing_duration - elapsed
                    print(f"  👁️  观赏阶段: {max(0, remaining):.1f}秒剩余")
        
        recorder.stop()
        self.auto_adjust_camera = original_auto_adjust
        self._original_camera_state = None
        print(f"\n✅ 视频已保存: {output_path}")

    def run_interactive(self, sim):
        """实时交互模式（有窗口）"""
        if not self.visible:
            print("警告：渲染器创建时 visible=False，无法交互。")
            return
        
        print("实时模式 - 鼠标拖拽旋转，滚轮缩放，按 ESC 退出")
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            
            # 可在此添加手动添加节点的逻辑，此处省略
            sim.update_physics()
            points = sim.get_positions_array()
            edges = sim.get_edge_vertices_array()
            
            if self.auto_adjust_camera:
                self._auto_adjust_camera_distance(points)
            
            self.update_buffers(points, edges)
            self._render_frame()
            glfw.swap_buffers(self.window)
        
        self.close()

    def close(self):
        glfw.terminate()