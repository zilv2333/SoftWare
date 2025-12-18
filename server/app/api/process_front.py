import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import json
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查MediaPipe版本
print(f"MediaPipe版本: {mp.__version__}")


class AdvancedPullUpBenchmark:
    def __init__(self, smooth_method='double_exponential', smooth_factor=0.7):
        """初始化，添加平滑方法参数"""
        # MediaPipe初始化
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # ============== 修改1: 使用MediaPipe 0.10.21兼容参数 ==============
        # 根据MediaPipe 0.10.21文档，正确的参数如下：
        try:
            # 先尝试最全的参数
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        except TypeError as e:
            # 如果还有问题，使用最简参数
            print(f"⚠️ 使用最简参数: {e}")
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        # ==================================================================

        # 关键点定义
        self.LANDMARK_INDICES = {
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26
        }

        self.BENCHMARK_POINTS = [0, 25, 50, 75, 100]

        # ============== 修改2: 添加平滑器初始化 ==============
        self.landmark_smoother = self.LandmarkSmoother(
            smooth_method=smooth_method,
            smoothing_factor=smooth_factor,
            filter_window=5
        )
        # =================================================

    # ============== 修改3: 添加LandmarkSmoother内部类 ==============
    class LandmarkSmoother:
        """专门用于MediaPipe关键点平滑的类"""

        def __init__(self, smooth_method='double_exponential',
                     smoothing_factor=0.7,
                     filter_window=5):
            self.method = smooth_method
            self.smoothing_factor = smoothing_factor
            self.window_size = filter_window

            # 存储历史数据
            self.history = []
            self.smoothed_history = []

        def smooth_frame(self, landmarks):
            """平滑单个帧的关键点"""
            if landmarks is None:
                return None

            # 提取关键点坐标
            current_points = self._extract_points(landmarks)

            # 添加到历史
            self.history.append(current_points)

            # 根据方法进行平滑
            if self.method == 'double_exponential':
                smoothed = self._double_exponential_smoothing(current_points)
            elif self.method == 'moving_average':
                smoothed = self._moving_average_smoothing(current_points)
            else:
                smoothed = current_points

            self.smoothed_history.append(smoothed)
            return self._create_landmarks(smoothed)

        def _double_exponential_smoothing(self, current_points):
            """双指数平滑 - 适用于有速度变化的运动"""
            if len(self.smoothed_history) == 0:
                return current_points

            last_smoothed = self.smoothed_history[-1]
            smoothed = {}

            for i, (x, y, z, v) in current_points.items():
                if i in last_smoothed:
                    # 位置平滑
                    s_x = self.smoothing_factor * x + (1 - self.smoothing_factor) * last_smoothed[i][0]
                    s_y = self.smoothing_factor * y + (1 - self.smoothing_factor) * last_smoothed[i][1]
                    s_z = self.smoothing_factor * z + (1 - self.smoothing_factor) * last_smoothed[i][2]

                    # 趋势平滑
                    if len(self.smoothed_history) > 1:
                        prev_smoothed = self.smoothed_history[-2]
                        trend_x = last_smoothed[i][0] - prev_smoothed[i][0]
                        trend_y = last_smoothed[i][1] - prev_smoothed[i][1]
                        trend_z = last_smoothed[i][2] - prev_smoothed[i][2]

                        s_x += self.smoothing_factor * trend_x
                        s_y += self.smoothing_factor * trend_y
                        s_z += self.smoothing_factor * trend_z
                else:
                    s_x, s_y, s_z = x, y, z

                smoothed[i] = (s_x, s_y, z, v)

            return smoothed

        def _moving_average_smoothing(self, current_points):
            """移动平均滤波"""
            if len(self.history) < 2:
                return current_points

            window = self.history[-self.window_size:] if len(self.history) >= self.window_size else self.history

            smoothed = {}

            for i in current_points.keys():
                # 收集窗口内该关键点的所有坐标
                points_in_window = []
                valid_frames = 0

                for frame in window:
                    if i in frame and len(frame[i]) >= 4 and frame[i][3] > 0.5:  # visibility > 0.5
                        points_in_window.append(frame[i])
                        valid_frames += 1

                if valid_frames > 0:
                    # 计算平均值
                    avg_x = sum(p[0] for p in points_in_window) / valid_frames
                    avg_y = sum(p[1] for p in points_in_window) / valid_frames
                    avg_z = sum(p[2] for p in points_in_window) / valid_frames

                    # 使用当前帧的可见度
                    visibility = current_points[i][3] if i in current_points and len(current_points[i]) >= 4 else 0.5
                    smoothed[i] = (avg_x, avg_y, avg_z, visibility)
                else:
                    smoothed[i] = current_points[i] if i in current_points else (0, 0, 0, 0)

            return smoothed

        def _extract_points(self, landmarks):
            """从MediaPipe Landmarks对象提取点"""
            points = {}
            for idx, landmark in enumerate(landmarks.landmark):
                points[idx] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
            return points

        def _create_landmarks(self, points_dict):
            """从点字典创建MediaPipe Landmarks对象"""

            class SimpleLandmark:
                def __init__(self, x, y, z, visibility):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility

            class SimpleLandmarkList:
                def __init__(self):
                    self.landmark = []

            landmark_list = SimpleLandmarkList()

            for i in sorted(points_dict.keys()):
                if i in points_dict and len(points_dict[i]) >= 4:
                    x, y, z, visibility = points_dict[i]
                    landmark_list.landmark.append(SimpleLandmark(x, y, z, visibility))

            return landmark_list

        def reset(self):
            """重置历史数据"""
            self.history = []
            self.smoothed_history = []

    # =============================================================

    def extract_comprehensive_landmarks(self, video_path, output_video_path=None, enable_smoothing=True):
        """提取综合关键点数据并生成简单可视化视频"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 视频写入器
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"🎬 将生成可视化视频: {output_video_path}")

        # 自定义躯干连接线
        TORSO_CONNECTIONS = [
            (15, 13), (16, 14), (13, 11), (14, 12),
            (11, 12), (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28)
        ]

        landmarks_data = []

        # ============== 修改4: 重置平滑器 ==============
        if enable_smoothing:
            self.landmark_smoother.reset()
        # =============================================

        with tqdm(total=total_frames, desc="提取综合关键点") as pbar:
            for frame_count in range(total_frames):
                success, frame = cap.read()
                if not success:
                    break

                display_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                frame_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps
                }

                if results.pose_landmarks:
                    # ============== 修改5: 应用平滑处理 ==============
                    if enable_smoothing:
                        smoothed_landmarks = self.landmark_smoother.smooth_frame(results.pose_landmarks)
                        # 使用平滑后的关键点进行绘制和计算
                        self._draw_custom_skeleton(display_frame, smoothed_landmarks, TORSO_CONNECTIONS, width, height)
                        frame_data.update(self._calculate_grip_metrics(smoothed_landmarks))
                        frame_data.update(self._calculate_height_metrics(smoothed_landmarks))
                        frame_data.update(self._calculate_torso_angle(smoothed_landmarks))
                    else:
                        # 使用原始关键点
                        self._draw_custom_skeleton(display_frame, results.pose_landmarks, TORSO_CONNECTIONS, width,
                                                   height)
                        frame_data.update(self._calculate_grip_metrics(results.pose_landmarks))
                        frame_data.update(self._calculate_height_metrics(results.pose_landmarks))
                        frame_data.update(self._calculate_torso_angle(results.pose_landmarks))
                    # ================================================

                    # 保存到视频文件
                    if out:
                        out.write(display_frame)

                else:
                    # 即使没有检测到关键点，也保存原始帧到视频
                    if out:
                        out.write(display_frame)

                    # 标记缺失数据
                    frame_data.update(self._get_nan_metrics())

                landmarks_data.append(frame_data)
                pbar.update(1)

        cap.release()

        # 关闭视频写入器
        if out:
            out.release()
            print(f"✅ 可视化视频已保存: {output_video_path}")

        return pd.DataFrame(landmarks_data)

    def _calculate_grip_metrics(self, landmarks):
        """计算握距相关指标"""
        metrics = {}
        try:
            # 使用世界坐标计算握距
            left_wrist = np.array([landmarks.landmark[15].x, landmarks.landmark[15].y])
            right_wrist = np.array([landmarks.landmark[16].x, landmarks.landmark[16].y])
            left_shoulder = np.array([landmarks.landmark[11].x, landmarks.landmark[11].y])
            right_shoulder = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])

            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

            metrics['GRIP_WIDTH'] = wrist_distance
            metrics['SHOULDER_WIDTH'] = shoulder_distance
            metrics['GRIP_RATIO'] = wrist_distance / shoulder_distance if shoulder_distance > 0 else np.nan

        except Exception as e:
            metrics['GRIP_WIDTH'] = np.nan
            metrics['SHOULDER_WIDTH'] = np.nan
            metrics['GRIP_RATIO'] = np.nan

        return metrics

    def _calculate_height_metrics(self, landmarks):
        """计算高度相关指标"""
        metrics = {}
        try:
            # 使用归一化坐标（0-1范围）
            # 手腕坐标
            left_wrist_x = landmarks.landmark[15].x
            left_wrist_y = landmarks.landmark[15].y
            right_wrist_x = landmarks.landmark[16].x
            right_wrist_y = landmarks.landmark[16].y

            left_shoulder_y = landmarks.landmark[11].y
            right_shoulder_y = landmarks.landmark[12].y
            # 添加肘部坐标
            left_elbow_x = landmarks.landmark[13].x
            left_elbow_y = landmarks.landmark[13].y
            right_elbow_x = landmarks.landmark[14].x
            right_elbow_y = landmarks.landmark[14].y

            metrics['LEFT_WRIST_X'] = left_wrist_x
            metrics['LEFT_WRIST_Y'] = left_wrist_y
            metrics['RIGHT_WRIST_X'] = right_wrist_x
            metrics['RIGHT_WRIST_Y'] = right_wrist_y

            metrics['LEFT_SHOULDER_Y'] = left_shoulder_y
            metrics['RIGHT_SHOULDER_Y'] = right_shoulder_y
            metrics['AVG_WRIST_HEIGHT'] = (left_wrist_y + right_wrist_y) / 2
            metrics['AVG_SHOULDER_HEIGHT'] = (left_shoulder_y + right_shoulder_y) / 2
            metrics['MIN_SHOULDER_HEIGHT'] = min(left_shoulder_y, right_shoulder_y)

            # 添加肘部坐标
            metrics['LEFT_ELBOW_X'] = left_elbow_x
            metrics['LEFT_ELBOW_Y'] = left_elbow_y
            metrics['RIGHT_ELBOW_X'] = right_elbow_x
            metrics['RIGHT_ELBOW_Y'] = right_elbow_y

        except Exception as e:
            metrics.update({key: np.nan for key in [
                'LEFT_WRIST_X', 'LEFT_WRIST_Y',  # 新增左手腕X坐标
                'RIGHT_WRIST_X', 'RIGHT_WRIST_Y',
                'LEFT_SHOULDER_Y', 'RIGHT_SHOULDER_Y',  # 新增右手腕X坐标'LEFT_SHOULDER_Y', 'RIGHT_SHOULDER_Y',
                'AVG_WRIST_HEIGHT', 'AVG_SHOULDER_HEIGHT', 'MIN_SHOULDER_HEIGHT', 'LEFT_ELBOW_X', 'LEFT_ELBOW_Y',
                'RIGHT_ELBOW_X', 'RIGHT_ELBOW_Y'
            ]})

        return metrics

    def _calculate_torso_angle(self, landmarks):
        """计算躯干角度"""
        metrics = {}
        try:
            # 肩膀中心
            left_shoulder = np.array([landmarks.landmark[11].x, landmarks.landmark[11].y])
            right_shoulder = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )

            # 髋部中心
            left_hip = np.array([landmarks.landmark[23].x, landmarks.landmark[23].y])
            right_hip = np.array([landmarks.landmark[24].x, landmarks.landmark[24].y])
            hip_center = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

            # 躯干向量
            dx = shoulder_center[0] - hip_center[0]  # 水平分量
            dy = shoulder_center[1] - hip_center[1]  # 垂直分量

            # 计算与垂直线的夹角
            angle = np.degrees(np.arctan2(dx, dy))
            metrics['TORSO_ANGLE'] = angle
            metrics['TORSO_ANGLE_ABS'] = abs(angle)  # 绝对值表示倾斜程度

        except Exception as e:
            metrics['TORSO_ANGLE'] = np.nan
            metrics['TORSO_ANGLE_ABS'] = np.nan

        return metrics

    def _draw_custom_skeleton(self, frame, landmarks, connections, width, height):
        """自定义绘制骨架"""
        # 绘制连接线
        for start_idx, end_idx in connections:
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]

            if start_landmark.visibility > 0.3 and end_landmark.visibility > 0.3:
                start_x = int(start_landmark.x * width)
                start_y = int(start_landmark.y * height)
                end_x = int(end_landmark.x * width)
                end_y = int(end_landmark.y * height)

                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

        # 绘制关键点
        connected_points = set()
        for connection in connections:
            connected_points.add(connection[0])
            connected_points.add(connection[1])

        for point_idx in connected_points:
            landmark = landmarks.landmark[point_idx]
            if landmark.visibility > 0.3:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def detect_rep_cycles_by_shoulder_height(self, df):
        """基于肩膀高度检测引体向上周期"""
        print("基于肩膀高度检测引体向上周期...")

        # 使用肩膀高度作为主要信号
        shoulder_heights = df['MIN_SHOULDER_HEIGHT'].values

        # 数据清理和插值
        shoulder_series = pd.Series(shoulder_heights)
        shoulder_interp = shoulder_series.interpolate(method='linear', limit_direction='both')

        if len(shoulder_interp) < 20:
            print("数据太少，无法检测周期")
            return []

        # 平滑信号
        window_size = min(11, len(shoulder_interp) // 10 * 2 + 1)
        if window_size < 3:
            window_size = 3

        try:
            smoothed = signal.savgol_filter(shoulder_interp, window_length=window_size, polyorder=2)
        except Exception as e:
            print(f"平滑信号失败: {e}")
            smoothed = shoulder_interp.values

        # 寻找周期
        rep_cycles = self._find_cycles_by_shoulder_height(smoothed)
        print(f"检测到 {len(rep_cycles)} 个引体向上周期")
        return rep_cycles

    def _find_cycles_by_shoulder_height(self, shoulder_heights):
        """基于肩膀高度寻找周期"""
        rep_cycles = []

        try:
            min_distance = max(15, len(shoulder_heights) // 20)

            # 寻找波谷（肩膀最高点）
            valleys, _ = signal.find_peaks(-shoulder_heights, distance=min_distance, prominence=0.02)
            # 寻找波峰（肩膀最低点）
            peaks, _ = signal.find_peaks(shoulder_heights, distance=min_distance, prominence=0.02)
            peaks = self._add_boundary_peaks(shoulder_heights, peaks, min_distance)
            print(f"肩膀高度检测: {len(peaks)}个波峰(手臂伸直), {len(valleys)}个波谷(下巴过杆)")

            # 构建周期
            if len(peaks) >= 2 and len(valleys) >= 1:
                for i in range(len(peaks) - 1):
                    start_peak = peaks[i]
                    end_peak = peaks[i + 1]

                    # 在两个波峰之间寻找波谷
                    valleys_between = [v for v in valleys if start_peak < v < end_peak]

                    if valleys_between:
                        valley = valleys_between[0]

                        if self._validate_rep_cycle(shoulder_heights, start_peak, valley, end_peak):
                            rep_cycles.append({
                                'start_frame': int(start_peak),
                                'bottom_frame': int(valley),
                                'end_frame': int(end_peak),
                                'duration': int(end_peak - start_peak),
                                'amplitude': float(shoulder_heights[start_peak] - shoulder_heights[valley])
                            })

        except Exception as e:
            print(f"肩膀高度周期检测错误: {e}")

        return rep_cycles

    def _add_boundary_peaks(self, signal_data, detected_peaks, min_distance):
        peaks = list(detected_peaks)

        # 检查起始边界（第一帧）
        if len(signal_data) > 0:
            search_range = min(min_distance, len(signal_data) // 4)
            if search_range > 0:
                first_value = signal_data[0]  # 第一帧的值
                subsequent_values = signal_data[1:search_range]  # 后续几帧

                # 条件1：第一帧 > 后续帧的最大值
                if len(subsequent_values) > 0 and first_value > np.max(subsequent_values):
                    # 条件2：第一帧 > 整个数据的60%分位数（确保是真正的高点）
                    if first_value > np.percentile(signal_data, 60):
                        peaks.insert(0, 0)  # 添加第一帧为波峰

        return np.array(sorted(peaks))

    def _validate_rep_cycle(self, signal_data, start, bottom, end):
        """验证周期有效性"""
        try:
            if end <= start or bottom <= start or end <= bottom:
                return False

            duration = end - start
            amplitude = signal_data[start] - signal_data[bottom]

            # 宽松的验证条件
            if duration < 10 or duration > 200 or amplitude < 0.02:
                return False

            return True
        except Exception as e:
            return False

    def create_biomechanical_benchmark(self, df, rep_cycles):
        """创建生物力学基准"""
        if not rep_cycles:
            print("没有检测到周期，创建空基准")
            return self._create_empty_benchmark()

        # 分析每个周期
        cycle_analyses = {}

        for i, cycle in enumerate(rep_cycles):
            cycle_name = f"cycle_{i + 1}"
            cycle_analysis = self._analyze_single_cycle(df, cycle, cycle_name)
            if cycle_analysis:
                cycle_analyses[cycle_name] = cycle_analysis

        if not cycle_analyses:
            return self._create_empty_benchmark()

        # 创建基准结果
        benchmark = {
            'analysis_summary': {
                'total_cycles': len(cycle_analyses),
                'total_frames': len(df),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'status': 'success'
            },
            'cycles': cycle_analyses
        }

        return benchmark

    def _analyze_single_cycle(self, df, cycle, cycle_name):
        """分析单个周期"""
        try:
            start, bottom, end = cycle['start_frame'], cycle['bottom_frame'], cycle['end_frame']

            if end >= len(df):
                return None

            cycle_data = df.iloc[start:end].copy()

            # 计算握距统计
            grip_ratios = cycle_data['GRIP_RATIO'].dropna()
            grip_stats = {
                'grip_ratio_mean': float(np.mean(grip_ratios)) if len(grip_ratios) > 0 else np.nan,
                'grip_ratio_max': float(np.max(grip_ratios)) if len(grip_ratios) > 0 else np.nan,
                'grip_ratio_min': float(np.min(grip_ratios)) if len(grip_ratios) > 0 else np.nan,
                'grip_ratio_std': float(np.std(grip_ratios)) if len(grip_ratios) > 0 else np.nan
            }

            # 计算躯干角度统计
            torso_angles = cycle_data['TORSO_ANGLE_ABS'].dropna()
            torso_stats = {
                'torso_angle_max': float(np.max(torso_angles)) if len(torso_angles) > 0 else np.nan,
                'torso_angle_min': float(np.min(torso_angles)) if len(torso_angles) > 0 else np.nan,
                'torso_angle_mean': float(np.mean(torso_angles)) if len(torso_angles) > 0 else np.nan,
                'torso_angle_std': float(np.std(torso_angles)) if len(torso_angles) > 0 else np.nan
            }

            # 计算最高点（下巴过杠点）的肩膀中心与手腕中心高度差
            peak_height_diff = self._calculate_peak_height_difference(cycle_data, bottom)

            # 计算最高点时手腕-肘部角度
            wrist_elbow_angle = self._calculate_wrist_elbow_angle_at_peak(cycle_data, bottom)

            cycle_analysis = {
                'cycle_info': {
                    'start_frame': int(start),
                    'bottom_frame': int(bottom),
                    'end_frame': int(end),
                    'duration_frames': int(end - start),
                    'amplitude': float(cycle['amplitude'])
                },
                'grip_metrics': grip_stats,
                'torso_metrics': torso_stats,
                'peak_height_difference': peak_height_diff,
                'wrist_elbow_angle': wrist_elbow_angle
            }

            return cycle_analysis

        except Exception as e:
            print(f"分析周期 {cycle_name} 错误: {e}")
            return None

    def _calculate_peak_height_difference(self, cycle_data, bottom_frame):
        """计算最高点（下巴过杠点）的肩膀中心与手腕中心高度差"""
        try:
            # 修正索引处理：找到cycle_data中距离bottom_frame最近的帧
            cycle_start = cycle_data.index[0]
            relative_bottom = bottom_frame - cycle_start

            # 确保索引在有效范围内
            if 0 <= relative_bottom < len(cycle_data):
                bottom_data = cycle_data.iloc[relative_bottom]

                # 获取高度数据
                shoulder_center_y = bottom_data.get('AVG_SHOULDER_HEIGHT', np.nan)
                wrist_center_y = bottom_data.get('AVG_WRIST_HEIGHT', np.nan)

                print(f"调试: 帧{bottom_frame} - 肩膀高度: {shoulder_center_y}, 手腕高度: {wrist_center_y}")  # 调试信息

                if not np.isnan(shoulder_center_y) and not np.isnan(wrist_center_y):
                    height_diff = shoulder_center_y - wrist_center_y
                    return {
                        'height_difference': float(height_diff),
                        'shoulder_height': float(shoulder_center_y),
                        'wrist_height': float(wrist_center_y),
                        'frame': int(bottom_frame)
                    }

            return {
                'height_difference': np.nan,
                'shoulder_height': np.nan,
                'wrist_height': np.nan,
                'frame': int(bottom_frame)
            }

        except Exception as e:
            print(f"计算高度差错误: {e}")
            return {
                'height_difference': np.nan,
                'shoulder_height': np.nan,
                'wrist_height': np.nan,
                'frame': int(bottom_frame)
            }

    def _calculate_wrist_elbow_angle_at_peak(self, cycle_data, bottom_frame):
        """计算最高点时手腕与手肘之间的连接向量与垂直方向的夹角"""
        try:
            # 修正索引处理：找到cycle_data中距离bottom_frame最近的帧
            cycle_start = cycle_data.index[0]
            relative_bottom = bottom_frame - cycle_start

            # 确保索引在有效范围内
            if 0 <= relative_bottom < len(cycle_data):
                bottom_data = cycle_data.iloc[relative_bottom]

                # 获取左手手腕和手肘坐标
                left_wrist_x = bottom_data.get('LEFT_WRIST_X')
                left_wrist_y = bottom_data.get('LEFT_WRIST_Y')
                left_elbow_x = bottom_data.get('LEFT_ELBOW_X')
                left_elbow_y = bottom_data.get('LEFT_ELBOW_Y')

                # 获取右手手腕和手肘坐标
                right_wrist_x = bottom_data.get('RIGHT_WRIST_X')
                right_wrist_y = bottom_data.get('RIGHT_WRIST_Y')
                right_elbow_x = bottom_data.get('RIGHT_ELBOW_X')
                right_elbow_y = bottom_data.get('RIGHT_ELBOW_Y')

                # 计算左手手腕-手肘向量与垂直方向的夹角
                left_angle = np.nan
                if not np.isnan(left_wrist_x) and not np.isnan(left_wrist_y) and \
                        not np.isnan(left_elbow_x) and not np.isnan(left_elbow_y):
                    # 手腕到手肘的向量
                    wrist_to_elbow_x = left_elbow_x - left_wrist_x
                    wrist_to_elbow_y = left_elbow_y - left_wrist_y

                    # 垂直方向向量 (0, 1) 向下为正
                    vertical_vector = np.array([0, -1])
                    wrist_elbow_vector = np.array([wrist_to_elbow_x, wrist_to_elbow_y])

                    # 计算夹角
                    if np.linalg.norm(wrist_elbow_vector) > 0:
                        cos_angle = np.dot(wrist_elbow_vector, vertical_vector) / \
                                    (np.linalg.norm(wrist_elbow_vector) * np.linalg.norm(vertical_vector))
                        # 限制cos值在[-1, 1]范围内
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        left_angle = np.degrees(np.arccos(cos_angle))

                # 计算右手手腕-手肘向量与垂直方向的夹角
                right_angle = np.nan
                if not np.isnan(right_wrist_x) and not np.isnan(right_wrist_y) and \
                        not np.isnan(right_elbow_x) and not np.isnan(right_elbow_y):
                    # 手腕到手肘的向量
                    wrist_to_elbow_x = right_elbow_x - right_wrist_x
                    wrist_to_elbow_y = right_elbow_y - right_wrist_y

                    # 垂直方向向量 (0, 1) 向下为正
                    vertical_vector = np.array([0, -1])
                    wrist_elbow_vector = np.array([wrist_to_elbow_x, wrist_to_elbow_y])

                    # 计算夹角
                    if np.linalg.norm(wrist_elbow_vector) > 0:
                        cos_angle = np.dot(wrist_elbow_vector, vertical_vector) / \
                                    (np.linalg.norm(wrist_elbow_vector) * np.linalg.norm(vertical_vector))
                        # 限制cos值在[-1, 1]范围内
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        right_angle = np.degrees(np.arccos(cos_angle))

                return {
                    'left_wrist_elbow_angle': float(left_angle) if not np.isnan(left_angle) else np.nan,
                    'right_wrist_elbow_angle': float(right_angle) if not np.isnan(right_angle) else np.nan,
                    'avg_wrist_elbow_angle': float(np.nanmean([left_angle, right_angle])) if not all(
                        np.isnan([left_angle, right_angle])) else np.nan,
                    'frame': int(bottom_frame)
                }

            return {
                'left_wrist_elbow_angle': np.nan,
                'right_wrist_elbow_angle': np.nan,
                'avg_wrist_elbow_angle': np.nan,
                'frame': int(bottom_frame)
            }

        except Exception as e:
            print(f"计算手腕-手肘角度错误: {e}")
            return {
                'left_wrist_elbow_angle': np.nan,
                'right_wrist_elbow_angle': np.nan,
                'avg_wrist_elbow_angle': np.nan,
                'frame': int(bottom_frame)
            }

    def _get_nan_metrics(self):
        """返回NaN指标字典"""
        return {
            'GRIP_WIDTH': np.nan, 'SHOULDER_WIDTH': np.nan, 'GRIP_RATIO': np.nan,
            'LEFT_WRIST_X': np.nan, 'LEFT_WRIST_Y': np.nan,  # 新增
            'RIGHT_WRIST_X': np.nan, 'RIGHT_WRIST_Y': np.nan,  # 新增
            'LEFT_SHOULDER_Y': np.nan, 'RIGHT_SHOULDER_Y': np.nan,
            'AVG_WRIST_HEIGHT': np.nan, 'AVG_SHOULDER_HEIGHT': np.nan, 'MIN_SHOULDER_HEIGHT': np.nan,
            'WRIST_SHOULDER_DIFF_LEFT': np.nan, 'WRIST_SHOULDER_DIFF_RIGHT': np.nan, 'WRIST_SHOULDER_DIFF_AVG': np.nan,
            'TORSO_ANGLE': np.nan, 'TORSO_ANGLE_ABS': np.nan,
            'LEFT_ELBOW_X': np.nan, 'LEFT_ELBOW_Y': np.nan, 'RIGHT_ELBOW_X': np.nan, 'RIGHT_ELBOW_Y': np.nan
        }

    def _create_empty_benchmark(self):
        """创建空基准"""
        return {
            'analysis_summary': {
                'total_cycles': 0,
                'total_frames': 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'status': 'no_cycles_detected'
            },
            'cycles': {}
        }

    # ============== 修改6: 添加后处理滤波方法 ==============
    def post_process_filtering(self, df, method='butterworth', order=4, cutoff_freq=0.1):
        """后处理滤波 - 在数据提取完成后进行更精细的平滑"""
        df_smoothed = df.copy()

        # 需要平滑的列
        coordinate_columns = [
            'LEFT_WRIST_X', 'LEFT_WRIST_Y',
            'RIGHT_WRIST_X', 'RIGHT_WRIST_Y',
            'LEFT_ELBOW_X', 'LEFT_ELBOW_Y',
            'RIGHT_ELBOW_X', 'RIGHT_ELBOW_Y',
            'LEFT_SHOULDER_Y', 'RIGHT_SHOULDER_Y',
            'GRIP_WIDTH', 'SHOULDER_WIDTH',
            'GRIP_RATIO', 'TORSO_ANGLE'
        ]

        fps = 30  # 估计的帧率，可以根据实际情况调整

        for col in coordinate_columns:
            if col in df.columns:
                series = df[col].copy()

                # 检查是否有有效数据
                if series.isna().all() or len(series) < 10:
                    continue

                # 插值填充缺失值
                series_filled = series.interpolate(method='linear', limit_direction='both')
                series_filled = series_filled.ffill().bfill()  # 前后填充

                if method == 'butterworth':
                    # 巴特沃斯滤波器 - 最适合生物信号
                    nyquist = fps / 2
                    normal_cutoff = cutoff_freq / nyquist

                    if normal_cutoff < 1.0:  # 确保截止频率有效
                        b, a = signal.butter(order, normal_cutoff, btype='low')

                        # 向前向后滤波（零相位失真）
                        filtered = signal.filtfilt(b, a, series_filled)

                        # 确保滤波后数据范围合理
                        if 'ANGLE' in col or 'GRIP' in col:
                            # 角度和比例值范围检查
                            filtered = np.clip(filtered, series_filled.min() * 0.5, series_filled.max() * 1.5)

                        df_smoothed[col] = filtered

                elif method == 'savgol':
                    # Savitzky-Golay滤波器
                    window_length = min(11, len(series_filled) // 3 * 2 + 1)  # 自动调整窗口
                    if window_length >= 5 and window_length <= len(series_filled):
                        polyorder = min(3, window_length - 1)
                        try:
                            filtered = signal.savgol_filter(
                                series_filled,
                                window_length=window_length,
                                polyorder=polyorder
                            )
                            df_smoothed[col] = filtered
                        except:
                            df_smoothed[col] = series_filled
                    else:
                        df_smoothed[col] = series_filled

        return df_smoothed
    # =====================================================

def process(front_path):
    benchmark_system = AdvancedPullUpBenchmark(
        smooth_method='double_exponential',  # 使用双指数平滑
        smooth_factor=0.7  # 平滑因子
    )
    i = 0

    df = benchmark_system.extract_comprehensive_landmarks(front_path)
    if df is not None:
        df_smoothed = benchmark_system.post_process_filtering(
            df,
            method='butterworth',  # 巴特沃斯滤波器
            cutoff_freq=0.1  # 截止频率（Hz），保留低频信号
        )

        # 使用平滑后的数据进行分析
        rep_cycles = benchmark_system.detect_rep_cycles_by_shoulder_height(df_smoothed)
        benchmark = benchmark_system.create_biomechanical_benchmark(df_smoothed, rep_cycles)

        # 打印结果摘要
        if benchmark['analysis_summary']['status'] == 'success':
            res=f'我一共做了{len(rep_cycles)}个引体向上,下面是我每个周期从正面看的描述：'
            # print(f"\n📊 分析摘要:")
            # print(f"   周期数: {benchmark['analysis_summary']['total_cycles']}")

            for cycle_name, cycle_data in benchmark['cycles'].items():
                i = i + 1
                print(f"\n{cycle_name}:")
                grip = cycle_data['grip_metrics']
                torso = cycle_data['torso_metrics']
                peak = cycle_data['peak_height_difference']
                wrist_angle = cycle_data['wrist_elbow_angle']
                if wrist_angle['avg_wrist_elbow_angle'] is not None and not np.isnan(
                        wrist_angle['avg_wrist_elbow_angle']):
                    wrist_angle_string=(f"   手腕-肘部角度: 左手={wrist_angle['left_wrist_elbow_angle']:.1f}°, "
                                        f"右手={wrist_angle['right_wrist_elbow_angle']:.1f}°, "
                                        f"平均={wrist_angle['avg_wrist_elbow_angle']:.1f}°")

                res=res+(f"第{i}个周期：我的握距相对肩宽比例为：平均={grip['grip_ratio_mean']:.3f},最大={grip['grip_ratio_max']:.3f}, "
                         f"最小={grip['grip_ratio_min']:.3f} ;我的脊柱相对竖直线角度为：最大={torso['torso_angle_max']:.1f}°,"
                         f"最小={torso['torso_angle_min']:.1f}°, 平均={torso['torso_angle_mean']:.1f}°"
                         f"在最高点时，我肩膀连线与手腕连线的高度差为{peak['height_difference']:.3f}。")+wrist_angle_string


            return res,len(rep_cycles)
        else:
            print("❌ 未检测到有效的引体向上周期")
            return None
    else:
        print("❌ 数据提取失败")
        return None

# 使用示例
if __name__ == "__main__":

    print(process('./4.mp4'))
    # ============== 修改7: 使用带平滑的初始化 ==============

    # ====================================================

