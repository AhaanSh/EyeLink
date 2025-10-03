# Eye Tracking Mouse Control with MediaPipe

This project implements a **real-time eye-tracking system** that allows users to control their mouse cursor using only their eyes. It leverages **MediaPipe Face Mesh** for iris and blink detection, **OpenCV** for video processing, and **PyAutoGUI** for system-level mouse control.

Created by Ahaan Shah <ahaansh@umich.edu>
---

## Features

* **Eye and Iris Tracking** – Uses MediaPipe Face Mesh to track the iris center with high accuracy.
* **Calibration Routine** – Guides users through multiple calibration points on screen to map pupil movement to screen coordinates.
* **Blink Detection** – Uses **Eye Aspect Ratio (EAR)** to detect blinks. Holding a blink for **5 seconds** exits the program.
* **Dead Zone Filtering** – Prevents jitter by ignoring micro-movements around the eye center.
* **Smoothing Filter** – Exponential moving average applied to pupil vectors for smoother cursor motion.

---

## Tech Stack

* **Python 3.8+**
* [**MediaPipe**](https://developers.google.com/mediapipe/) – Eye and iris tracking
* [**OpenCV**](https://opencv.org/) – Video capture and visualization
* [**NumPy**](https://numpy.org/) – Vector math and smoothing
* [**PyAutoGUI**](https://pyautogui.readthedocs.io/en/latest/) – Cursor control

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/eye-tracking-mouse.git
cd eye-tracking-mouse
pip install mediapipe opencv-python pyautogui numpy
```

---

## Usage

### 1. Run the script

```bash
python eye_mouse.py
```

### 2. Calibration

* The program will guide you to look at **5 calibration points** (corners + center).
* Keep your head steady while focusing on each target until calibration completes.

### 3. Control

* Once calibrated, your **eye movements** will map directly to screen cursor movements.
* A **dead zone** prevents small tremors from moving the cursor.

### 4. Exit

* Hold a **blink for 5 seconds** *or* press `ESC` to quit.

---


