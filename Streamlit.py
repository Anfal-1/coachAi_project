import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import pygame
import torch
import pickle
from pathlib import Path
import tempfile
import json
import os

st.set_page_config(
    page_title="🏋️ AI Posture Trainer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Session Storage =====
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'issues' not in st.session_state:
    st.session_state.issues = []
if 'quality' not in st.session_state:
    st.session_state.quality = 0


# ===== Load Models =====
@st.cache_resource
def load_yolo_model():
    """Load YOLOv5 model"""
    try:
        model_weights_path = "./models/best_big_bounding.pt"
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLOv5: {str(e)}")
        return None


@st.cache_resource
def load_mediapipe_pose():
    """Load MediaPipe Pose model"""
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        return pose, mp_pose
    except Exception as e:
        st.error(f"❌ Error loading MediaPipe: {str(e)}")
        return None, None


# Load models
model_yolo = load_yolo_model()
pose, mp_pose = load_mediapipe_pose()


# ===== Calculation Functions =====
def calculateAngle(a, b, c):
    """Calculate angle between three points"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except:
        return 0


def detect_objects(frame):
    """Detect people using YOLOv5"""
    if model_yolo is None:
        return None
    try:
        results = model_yolo(frame)
        pred = results.pred[0] if results.pred is not None else None
        return pred
    except:
        return None


def extract_landmarks(results_pose, frame_height, frame_width):
    """Extract body keypoints"""
    landmarks = {}
    if results_pose and results_pose.pose_landmarks:
        for idx, landmark in enumerate(mp_pose.PoseLandmark):
            try:
                lm = results_pose.pose_landmarks.landmark[idx]
                landmarks[landmark.name] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
            except:
                pass
    return landmarks


def calculate_biomechanics(landmarks):
    """Calculate movement metrics"""
    metrics = {}

    try:
        if not landmarks:
            return metrics

        # Left side
        if 'LEFT_SHOULDER' in landmarks and 'LEFT_ELBOW' in landmarks and 'LEFT_WRIST' in landmarks:
            left_shoulder = [landmarks['LEFT_SHOULDER']['x'], landmarks['LEFT_SHOULDER']['y']]
            left_elbow = [landmarks['LEFT_ELBOW']['x'], landmarks['LEFT_ELBOW']['y']]
            left_wrist = [landmarks['LEFT_WRIST']['x'], landmarks['LEFT_WRIST']['y']]
            metrics['left_elbow'] = calculateAngle(left_shoulder, left_elbow, left_wrist)

        # Right side
        if 'RIGHT_SHOULDER' in landmarks and 'RIGHT_ELBOW' in landmarks and 'RIGHT_WRIST' in landmarks:
            right_shoulder = [landmarks['RIGHT_SHOULDER']['x'], landmarks['RIGHT_SHOULDER']['y']]
            right_elbow = [landmarks['RIGHT_ELBOW']['x'], landmarks['RIGHT_ELBOW']['y']]
            right_wrist = [landmarks['RIGHT_WRIST']['x'], landmarks['RIGHT_WRIST']['y']]
            metrics['right_elbow'] = calculateAngle(right_shoulder, right_elbow, right_wrist)

        # Hip and knees
        if 'LEFT_HIP' in landmarks and 'LEFT_KNEE' in landmarks and 'LEFT_ANKLE' in landmarks:
            left_hip = [landmarks['LEFT_HIP']['x'], landmarks['LEFT_HIP']['y']]
            left_knee = [landmarks['LEFT_KNEE']['x'], landmarks['LEFT_KNEE']['y']]
            left_ankle = [landmarks['LEFT_ANKLE']['x'], landmarks['LEFT_ANKLE']['y']]
            metrics['left_knee'] = calculateAngle(left_hip, left_knee, left_ankle)

        if 'RIGHT_HIP' in landmarks and 'RIGHT_KNEE' in landmarks and 'RIGHT_ANKLE' in landmarks:
            right_hip = [landmarks['RIGHT_HIP']['x'], landmarks['RIGHT_HIP']['y']]
            right_knee = [landmarks['RIGHT_KNEE']['x'], landmarks['RIGHT_KNEE']['y']]
            right_ankle = [landmarks['RIGHT_ANKLE']['x'], landmarks['RIGHT_ANKLE']['y']]
            metrics['right_knee'] = calculateAngle(right_hip, right_knee, right_ankle)

    except Exception as e:
        st.warning(f"⚠️ Warning in calculations: {str(e)}")

    return metrics


def analyze_posture_quality(metrics):
    """Analyze posture quality"""
    issues = []

    if not metrics:
        return issues, 10

    try:
        # Analyze angles
        if metrics.get('left_elbow', 0) > 150:
            issues.append({
                'type': 'excessive_elbow_extension',
                'severity': 'moderate',
                'description': '⚠️ Excessive left arm extension'
            })

        if metrics.get('right_elbow', 0) > 150:
            issues.append({
                'type': 'excessive_elbow_extension',
                'severity': 'moderate',
                'description': '⚠️ Excessive right arm extension'
            })

        if metrics.get('left_knee', 0) < 60:
            issues.append({
                'type': 'knee_too_bent',
                'severity': 'moderate',
                'description': '⚠️ Left knee is too bent'
            })

    except Exception as e:
        st.warning(f"⚠️ Error in analysis: {str(e)}")

    quality_score = max(0, 10 - len(issues) * 1.5)
    return issues, quality_score


# ===== Main Interface =====
st.title("🏋️ Smart Posture Trainer")

with st.sidebar:
    st.header("⚙️ Settings")

    input_mode = st.radio(
        "📹 Choose source:",
        ["📷 Webcam (Live)", "📁 Upload Video"]
    )

    exercise_type = st.selectbox(
        "🏋️ Exercise Type:",
        ["Bench Press", "Squat", "Deadlift"]
    )

    confidence_threshold = st.slider(
        "🎯 Confidence Threshold:",
        0.0, 1.0, 0.7, 0.05
    )

# ===== Content =====
if input_mode == "📷 Webcam (Live)":
    st.header("📹 Live Webcam Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("📊 Metrics")
        metrics_placeholder = st.empty()
        issues_placeholder = st.empty()
        quality_placeholder = st.empty()

    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)

    if start_btn and model_yolo and pose:
        st.info("📹 Starting camera capture...")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            st.error("❌ Cannot open camera. Ensure camera is connected.")
        else:
            frame_count = 0

            while not stop_btn and frame_count < 300:  # Maximum 300 frames
                ret, frame = camera.read()

                if not ret:
                    st.error("❌ Error reading video")
                    break

                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)

                    # Detect people
                    results_yolo = detect_objects(frame)

                    st.session_state.metrics = {}
                    st.session_state.issues = []
                    st.session_state.quality = 0

                    if results_yolo is not None and len(results_yolo) > 0:
                        det = results_yolo[0]
                        c1, c2 = det[:2].int(), det[2:4].int()
                        conf = det[4]

                        if conf >= 0.5:
                            c1 = (int(c1[0]), int(c1[1]))
                            c2 = (int(c2[0]), int(c2[1]))

                            object_frame = frame[c1[1]:c2[1], c1[0]:c2[0]].copy()

                            results_pose = pose.process(cv2.cvtColor(object_frame, cv2.COLOR_RGB2BGR))

                            if results_pose and results_pose.pose_landmarks:
                                landmarks = extract_landmarks(results_pose, object_frame.shape[0],
                                                              object_frame.shape[1])
                                st.session_state.metrics = calculate_biomechanics(landmarks)
                                st.session_state.issues, st.session_state.quality = analyze_posture_quality(
                                    st.session_state.metrics)

                                # Draw landmarks
                                try:
                                    mp.solutions.drawing_utils.draw_landmarks(
                                        object_frame,
                                        results_pose.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                                    )
                                except:
                                    pass

                                frame[c1[1]:c2[1], c1[0]:c2[0]] = object_frame

                    # Display video - FIXED: استخدام width="stretch"
                    frame_placeholder.image(frame, width="stretch")

                    # Display metrics
                    if st.session_state.metrics:
                        metrics_text = "**Metrics:**\n\n"
                        for key, value in st.session_state.metrics.items():
                            metrics_text += f"• {key}: {value:.1f}°\n"
                        metrics_placeholder.markdown(metrics_text)

                    # Display issues
                    if st.session_state.issues:
                        issues_text = "**Detected Issues:**\n\n"
                        for issue in st.session_state.issues:
                            issues_text += f"- {issue['description']}\n"
                        issues_placeholder.warning(issues_text)
                    else:
                        issues_placeholder.success("✅ No issues detected!")

                    # Display quality score
                    quality_placeholder.metric("Movement Quality", f"{st.session_state.quality:.1f}/10")

                    frame_count += 1
                    time.sleep(0.01)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    break

            camera.release()
            st.success("✅ Analysis complete!")

else:
    st.header("📁 Video File Analysis")

    uploaded_file = st.file_uploader(
        "📤 Choose a video file:",
        type=["mp4", "avi", "mov"],
    )

    if uploaded_file and model_yolo and pose:
        temp_video_path = f"temp_video_{int(time.time())}.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.video(temp_video_path)

        if st.button("🔍 Start Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            camera = cv2.VideoCapture(temp_video_path)
            total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            all_metrics = []

            results_container = st.container()

            while True:
                ret, frame = camera.read()
                if not ret:
                    break

                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results_yolo = detect_objects(frame)

                    if results_yolo is not None and len(results_yolo) > 0:
                        det = results_yolo[0]
                        c1, c2 = det[:2].int(), det[2:4].int()
                        conf = det[4]

                        if conf >= 0.5:
                            c1 = (int(c1[0]), int(c1[1]))
                            c2 = (int(c2[0]), int(c2[1]))

                            object_frame = frame[c1[1]:c2[1], c1[0]:c2[0]].copy()
                            results_pose = pose.process(cv2.cvtColor(object_frame, cv2.COLOR_RGB2BGR))

                            if results_pose and results_pose.pose_landmarks:
                                landmarks = extract_landmarks(results_pose, object_frame.shape[0],
                                                              object_frame.shape[1])
                                metrics = calculate_biomechanics(landmarks)
                                all_metrics.append(metrics)

                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"⏳ Analyzing... {frame_count}/{total_frames}")

                except Exception as e:
                    st.warning(f"⚠️ Warning: {str(e)}")

            camera.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")

            # Display results
            if all_metrics:
                st.success(f"✅ Analyzed {frame_count} frames")

        # Delete temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)