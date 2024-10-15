import streamlit as st
import numpy as np
import torch
from pygame import mixer
import cv2

# Load YOLOv5 model
# 'custom' indicates that we are loading a custom-trained model from a local path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"C:\Users\gokul\PycharmProjects\driver_drowsiness\last.pt", force_reload=True)

# Initialize the mixer for playing sounds
mixer.init()
sound = mixer.Sound(r"C:\Users\gokul\PycharmProjects\driver_drowsiness\alram.wav")

# Function to start the camera and process frames
def start_camera():
    global score
    score = 0  # Initialize the score variable to track drowsiness detection

    cap = cv2.VideoCapture(0)  # Start capturing video from the webcam

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the camera

        # Make detections using the YOLOv5 model
        results = model(frame)

        # Extract height and width of the frame for display purposes
        height, width = frame.shape[0:2]
        class_id = 16  # Assuming class_id 16 corresponds to a detected drowsy state

        # Check if the drowsiness class is detected in the results
        if class_id in results.pred[0][:, 5]:
            # Increment the score variable if drowsiness is detected
            score += 1
            if score > 15:  # Play sound if score exceeds 15
                try:
                    sound.play()  # Play alarm sound
                except:
                    pass
        else:
            # Print the detected class ids for debugging
            print((results.pred[0][:, 5]))
            score -= 1  # Decrease score if no drowsiness is detected
            if score < 0:
                score = 0  # Ensure score does not go negative

        # Display the current score on the video frame
        cv2.putText(frame, 'Score: ' + str(score), (100, height - 20),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        # Render detections on the frame
        rendered = np.squeeze(results.render())
        frame[0:rendered.shape[0], 0:rendered.shape[1]] = rendered

        # Show the video frame with detections
        cv2.imshow('YOLO', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Streamlit App
st.title("Drowsiness Detection with YOLOv5")

# Button to start the camera in the Streamlit app
if st.button("Start Camera"):
    start_camera()
