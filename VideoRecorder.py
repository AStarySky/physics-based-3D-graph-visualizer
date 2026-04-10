import cv2
import numpy as np
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE, glReadBuffer, GL_FRONT
from typing import Optional
import threading
from collections import deque
from enum import Enum
import time

class RenderPhase(Enum):
    GROWTH = "growth"
    VIEWING = "viewing"
    FINISHED = "finished"

class AutoVideoRecorder:
    """自动视频录制器 - 生长阶段由节点数控制"""
    
    def __init__(self, fps: int = 60, codec: str = 'mp4v'):
        self.fps = fps
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.width = 0
        self.height = 0
        
        # 异步写入队列
        self.frame_queue = deque(maxlen=200)
        self.writer_thread: Optional[threading.Thread] = None
        self.stop_writer = False
        
        # 渲染阶段控制
        self.phase = RenderPhase.GROWTH
        self.phase_start_time = 0.0
        self.frame_count = 0
        self.total_frames = 0
        
        # 生长控制（基于节点数）
        self.total_nodes = 0
        self.current_nodes = 0
        self.nodes_per_frame = 2
        
        # 观赏阶段参数
        self.viewing_duration = 8.0
        self.rotation_speed = 15.0
        
        # 状态
        self.paused = False
        self.current_rotation = 0.0
        self.growth_complete = False
        
    def start_auto_render(self, filename: str, width: int, height: int,
                          total_nodes: int,
                          viewing_duration: float = 8.0,
                          nodes_per_frame: int = 2):
        """
        开始自动渲染
        
        Args:
            filename: 输出文件名（自动添加 .mp4）
            width, height: 视频分辨率
            total_nodes: 图的总节点数（用于计算生长进度）
            viewing_duration: 观赏阶段持续时间（秒）
            nodes_per_frame: 每帧添加的节点数（控制生长速度）
        """
        if not filename.endswith('.mp4'):
            filename += '.mp4'
            
        self.width = width
        self.height = height
        self.total_nodes = total_nodes
        self.viewing_duration = viewing_duration
        self.nodes_per_frame = nodes_per_frame
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {filename}")
        
        # 重置状态
        self.phase = RenderPhase.GROWTH
        self.phase_start_time = time.time()
        self.frame_count = 0
        self.total_frames = 0
        self.current_rotation = 1.0
        self.current_nodes = 0
        self.growth_complete = False
        self.paused = False
        self.is_recording = True
        
        # 启动写入线程
        self.stop_writer = False
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.start()
        
        print(f"\n🎬 开始自动渲染:")
        print(f"   📹 输出: {filename}")
        print(f"   📐 分辨率: {width}x{height} @ {self.fps}fps")
        print(f"   🌱 生长阶段: 共 {total_nodes} 个节点 ({nodes_per_frame} 节点/帧)")
        print(f"   👁️  观赏阶段: {viewing_duration}秒 (旋转 {self.rotation_speed}°/秒)")
        print()
        
    def update(self, delta_time: float, current_node_count: int) -> dict:
        """
        更新录制状态（每帧调用）
        
        Args:
            delta_time: 上一帧到当前帧的时间间隔（秒）
            current_node_count: 当前已激活的节点数
        
        Returns:
            包含当前阶段、进度等信息的字典
        """
        if not self.is_recording or self.paused:
            return self._get_status()
        
        self.current_nodes = current_node_count
        
        if self.phase == RenderPhase.GROWTH:
            # 当所有节点都已激活时切换到观赏阶段
            if self.current_nodes >= self.total_nodes and not self.growth_complete:
                self._switch_to_viewing()
                
        elif self.phase == RenderPhase.VIEWING:
            self.viewing_duration -= delta_time
            if self.viewing_duration < 0:
                self._finish_rendering()
                
        return self._get_status()
    
    def _switch_to_viewing(self):
        self.phase = RenderPhase.VIEWING
        self.phase_start_time = time.time()
        self.current_rotation = 0.0
        self.growth_complete = True
        print(f"\n✨ 生长阶段完成 (已激活 {self.current_nodes}/{self.total_nodes} 个节点)")
        print(f"   📊 生长阶段帧数: {self.frame_count}")
        print(f"   🔄 进入观赏阶段 ({self.viewing_duration}秒)...")
        
    def _finish_rendering(self):
        self.phase = RenderPhase.FINISHED
        self.is_recording = False
        self.stop_writer = True
        print(f"\n🎉 渲染完成!")
        print(f"   📊 总帧数: {self.total_frames}")
        print(f"   ⏱️  实际时长: {self.total_frames / self.fps:.1f}秒")
        
    def should_add_nodes(self) -> bool:
        """当前帧是否应该添加新节点"""
        return (self.phase == RenderPhase.GROWTH and 
                not self.paused and 
                self.is_recording and
                self.current_nodes < self.total_nodes)
    
    
    def _get_status(self) -> dict:
        current_time = time.time()
        if self.phase == RenderPhase.GROWTH:
            progress = self.current_nodes / self.total_nodes if self.total_nodes > 0 else 0.0
            elapsed = current_time - self.phase_start_time
            total_elapsed = elapsed
            total_duration = 0.0  # 未知
        elif self.phase == RenderPhase.VIEWING:
            progress = 1.0
            elapsed = current_time - self.phase_start_time
            total_elapsed = elapsed
            total_duration = self.viewing_duration
        else:
            progress = 1.0
            total_elapsed = 0.0
            total_duration = 0.0
            
        return {
            'phase': self.phase,
            'progress': progress,
            'elapsed': total_elapsed,
            'total_duration': total_duration,
            'frame_count': self.frame_count,
            'total_frames': self.total_frames,
            'rotation': self.current_rotation,
            'nodes': f"{self.current_nodes}/{self.total_nodes}"
        }
    
    def capture_frame(self):
        """从 OpenGL 帧缓冲区捕获一帧并加入队列"""
        if not self.is_recording or self.paused:
            return
            
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = cv2.flip(frame, 0)                       # 垂直翻转
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # 转为 BGR
        
        self._add_overlay(frame)
        
        if len(self.frame_queue) < self.frame_queue.maxlen - 1:
            self.frame_queue.append(frame)
            self.frame_count += 1
            self.total_frames += 1
    
    def _add_overlay(self, frame: np.ndarray):
        """在帧上绘制信息叠加层"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        status = self._get_status()
        if self.phase == RenderPhase.GROWTH:
            phase_text = f"GROWTH ({self.current_nodes}/{self.total_nodes} nodes)"
            color = (0, 255, 0)
        elif self.phase == RenderPhase.VIEWING:
            remaining = self.viewing_duration - (time.time() - self.phase_start_time)
            phase_text = f"VIEWING ({max(0, remaining):.1f}s remaining)"
            color = (255, 165, 0)
        else:
            phase_text = "FINISHED"
            color = (0, 255, 255)
            
        cv2.putText(frame, phase_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        bar_width = int(w * 0.3)
        bar_x = w - bar_width - 30
        cv2.rectangle(frame, (bar_x, 10), (bar_x + bar_width, 20), (100, 100, 100), 1)
        cv2.rectangle(frame, (bar_x, 10), 
                        (bar_x + int(bar_width * status['progress']), 20), color, -1)
        progress_text = f"{status['progress']*100:.0f}%"
        cv2.putText(frame, progress_text, (bar_x + bar_width + 5, 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frame_text = f"Frame: {self.total_frames}"
        cv2.putText(frame, frame_text, (w - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
    
    def _writer_loop(self):
        """后台线程：将队列中的帧写入视频文件"""
        while not self.stop_writer or self.frame_queue:
            if self.frame_queue:
                frame = self.frame_queue.popleft()
                if self.writer:
                    self.writer.write(frame)
            else:
                time.sleep(0.001)
    
    def stop(self):
        """停止录制并释放资源"""
        self.is_recording = False
        self.stop_writer = True
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
        while self.frame_queue:
            frame = self.frame_queue.popleft()
            if self.writer:
                self.writer.write(frame)
        if self.writer:
            self.writer.release()
            self.writer = None