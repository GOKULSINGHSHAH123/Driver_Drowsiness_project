import streamlit as st
import numpy as np
import torch
from pygame import mixer
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"C:\Users\gokul\PycharmProjects\driver_drowsiness\last.pt", force_reload=True)

# Initialize the mixer
mixer.init()
sound = mixer.Sound(r"C:\Users\gokul\PycharmProjects\driver_drowsiness\alram.wav")

# Initialize the score variable

# Function to start the camera and process frames
def start_camera():
    global score
    score = 0

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        # Make detections
        results = model(frame)

        height, width = frame.shape[0:2]  # Extract height and width
        class_id = 16
        if class_id in results.pred[0][:, 5]:
            # Increment the score variable
            score += 1
            if score > 15:
                try:
                    sound.play()
                except:
                    pass

        else:
            print((results.pred[0][:, 5]))
            score -= 1
            if score < 0:
                score = 0

                # Display the score on the frame
        cv2.putText(frame, 'Score: ' + str(score), (100, height - 20),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        rendered = np.squeeze(results.render())
        frame[0:rendered.shape[0], 0:rendered.shape[1]] = rendered

        cv2.imshow('YOLO', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
# Streamlit App
st.title("Drowsiness Detection with YOLOv5")

# Button to start the camera
if st.button("Start Camera"):
    start_camera()
