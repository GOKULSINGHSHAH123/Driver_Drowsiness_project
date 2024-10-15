# Drowsiness Detection with YOLOv5

This project implements a real-time drowsiness detection system using a custom YOLOv5 model. The system captures video from a webcam and detects whether the user is drowsy, playing an alarm sound if drowsiness is detected.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- Real-time drowsiness detection using YOLOv5.
- Visual feedback with a score indicating drowsiness levels.
- Alarm sound played when drowsiness is detected.

## Requirements

To run this project, you will need:

- Python 3.6 or later
- Streamlit
- NumPy
- PyTorch
- OpenCV
- Pygame

You can install the required packages using pip:

```bash
pip install streamlit numpy torch opencv-python pygame
Installation
Clone this repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Download the YOLOv5 model weights and place them in the project directory.

Ensure you have the required audio file (alarm sound) in the project directory.

Usage
Start the Streamlit app:

bash

streamlit run <script-name>.py
Open the app in your web browser (usually at http://localhost:8501).

Click on the Start Camera button to begin drowsiness detection.

To stop the camera, press 'q' or close the video window.
