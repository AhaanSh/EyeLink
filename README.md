Eye Tracking Mouse Control with MediaPipe

Created by Ahaan Shah <ahaansh@umich.edu>

This project implements a real-time eye-tracking system that allows users to control their mouse cursor using only their eyes. It leverages MediaPipe Face Mesh for iris and blink detection, OpenCV for video processing, and PyAutoGUI for system-level mouse control.

Features
- Eye and Iris Tracking: Uses MediaPipe Face Mesh to track the iris center with high accuracy.
Calibration Routine: Guides users through multiple calibration points on screen to map pupil movement to screen coordinates.
- Blink Detection: Uses Eye Aspect Ratio (EAR) to detect blinks. Holding a blink for 5 seconds exits the program.
- Dead Zone Filtering: Prevents jitter by ignoring micro-movements around the eye center.
Smoothing Filter: Exponential moving average applied to pupil vectors for smoother cursor motion.

Tech Stack
- Python 3.8+
- MediaPipe (eye and iris tracking)
- OpenCV (video capture and visualization)
- NumPy (vector math and smoothing)
- PyAutoGUI (cursor control)


Installation
Clone this repository and install dependencies:

git clone https://github.com/yourusername/eye-tracking-mouse.git
cd eye-tracking-mouse
pip install mediapipe opencv-python pyautogui numpy

Usage
Run the script:

python eye_mouse.py


Calibration:
The program will guide you to look at 5 calibration points (corners + center).
Keep your head steady while focusing on each target until calibration completes.

Control:
Once calibrated, your eye movements will map directly to screen cursor movements.
A dead zone prevents small tremors from moving the cursor.

Exit:
Hold a blink for 5 seconds OR press ESC to quit.
