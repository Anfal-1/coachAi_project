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

st.set_page_config(
    page_title="Real-time AI Exercise Posture Correction System",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Load YOLOv5 model
model_weights_path = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.to("mps")
model.eval()

# Record previous alert time
previous_alert_time = 0


def most_frequent(data):
    return max(data, key=data.count)


# Angle calculation function
def calculateAngle(a, b, c):
    a = np.array(a)  # first point
    b = np.array(b)  # middle point
    c = np.array(c)  # end point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Object detection using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]
    return pred


# Initialize Streamlit app
st.title("Real-time AI Exercise Posture Correction System")

pygame.mixer.init()

# Sidebar exercise selection
menu_selection = st.selectbox(
    "Select Exercise",
    ("Bench Press", "Squat", "Deadlift")
)

counter_display = st.sidebar.empty()

counter = 0
counter_display.header(f"Current Counter: {counter} reps")

current_stage = ""
posture_status = [None]

model_weights_path = "./models/benchpress/benchpress.pkl"

with open(model_weights_path, "rb") as f:
    model_e = pickle.load(f)

if menu_selection == "Bench Press":
    model_weights_path = "./models/benchpress/benchpress.pkl"
elif menu_selection == "Squat":
    model_weights_path = "./models/squat/squat.pkl"
elif menu_selection == "Deadlift":
    model_weights_path = "./models/deadlift/deadlift.pkl"

with open(model_weights_path, "rb") as f:
    model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=2
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Landmark Detection Confidence Threshold", 0.0, 1.0, 0.7
)

# Angle display placeholders
neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # Run YOLOv5 detection
    results_yolo = detect_objects(frame)

    try:
        if results_yolo is not None:
            for det in results_yolo:
                c1, c2 = det[:2].int(), det[2:4].int()
                cls, conf, *_ = det

                if conf >= 0.7:

                    c1 = (c1[0].item(), c1[1].item())
                    c2 = (c2[0].item(), c2[1].item())

                    object_frame = frame[c1[1]:c2[1], c1[0]:c2[0]]
                    object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)

                    results_pose = pose.process(object_frame_rgb)

                    if results_pose.pose_landmarks is not None:

                        landmarks = results_pose.pose_landmarks.landmark

                        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                                landmarks[mp_pose.PoseLandmark.NOSE].y]

                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]

                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]

                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]

                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]

                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

                        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]

                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                        # Angle calculations
                        neck_angle = (
                            calculateAngle(left_shoulder, nose, left_hip)
                            + calculateAngle(right_shoulder, nose, right_hip) / 2
                        )

                        # Display angles
                        neck_angle_display.text(f"Neck Angle: {neck_angle:.2f}°")

    except Exception:
        pass

    FRAME_WINDOW.image(frame)
