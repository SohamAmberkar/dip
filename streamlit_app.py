import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from imutils.video import VideoStream
from imutils import face_utils
import dlib
from scipy.spatial import distance as dist

# Load the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Display Video Stream
def video_stream():
    vs = VideoStream(src=0).start()
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 20
    COUNTER = 0
    TOTAL = 0

    stframe = st.empty()
    
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (600, 400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw green contours around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    cv2.putText(frame, "Disengaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                cv2.putText(frame, "Engaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR: {ear:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Total Disengaged: {TOTAL}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

# Display Screenshots
def view_screenshots():
    screenshot_path = "screenshots"
    if os.path.exists(screenshot_path):
        images = os.listdir(screenshot_path)
        if images:
            selected_image = st.selectbox("Select a Screenshot", images)
            image_path = os.path.join(screenshot_path, selected_image)
            st.image(image_path)
        else:
            st.write("No screenshots available.")
    else:
        st.write("Screenshot directory not found.")

# Display CSV Report
def view_report():
    if os.path.exists('engagement_report.csv'):
        df = pd.read_csv('engagement_report.csv')
        st.write(df)
    else:
        st.write("Engagement report not found.")

# Streamlit Sidebar Navigation
st.sidebar.title("Engagement Detection System")
option = st.sidebar.selectbox("Choose an option", ["Video Stream", "View Screenshots", "View Report"])

if option == "Video Stream":
    video_stream()
elif option == "View Screenshots":
    view_screenshots()
elif option == "View Report":
    view_report()
