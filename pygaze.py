#!/usr/bin/env python3
"""
Gaze Tracking with Linear Calibration
- Uses MediaPipe FaceMesh to track left iris center
- Calibrates with 5 screen points (corners + center)
- Maps webcam pupil coords -> screen coords using np.polyfit
- Toggle mouse control with 'M', calibrate with 'C'
- Reduced sensitivity, improved smoothing, jump limiting
"""

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import argparse
from collections import deque

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [468]  # just left iris center

class GazeTracker:
    def __init__(self, smoothing_window=5, frames_per_point=60, damping=0.5, max_jump=30):
        self.screen_w, self.screen_h = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibration data
        self.calib_cam_x, self.calib_cam_y = [], []
        self.calib_scr_x, self.calib_scr_y = [], []
        self.coef_x, self.coef_y = None, None

        self.frames_per_point = frames_per_point
        self.smoothing_window = smoothing_window
        self.gaze_history = deque(maxlen=smoothing_window)

        # Flags
        self.calibrated = False
        self.enable_mouse_control = False

        # Damping & jump limits
        self.damping = damping
        self.max_jump = max_jump
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2

        # 5 calibration points (screen coords)
        self.calib_points = [
            (100, 100, "top-left"),
            (self.screen_w-100, 100, "top-right"),
            (self.screen_w-100, self.screen_h-100, "bottom-right"),
            (100, self.screen_h-100, "bottom-left"),
            (self.screen_w//2, self.screen_h//2, "center")
        ]

    def get_pupil_center(self, face_landmarks, w, h):
        L = face_landmarks.landmark[LEFT_IRIS[0]]
        return np.array([L.x * w, L.y * h]).astype(int)

    def smooth_coords(self, coords):
        self.gaze_history.append(coords)
        xs, ys = zip(*self.gaze_history)
        return int(np.mean(xs)), int(np.mean(ys))

    def calibrate(self):
        print("Starting calibration...")
        step = 0
        frames_collected = 0
        pupil_x_list, pupil_y_list = [], []

        self.calib_cam_x.clear()
        self.calib_cam_y.clear()
        self.calib_scr_x.clear()
        self.calib_scr_y.clear()

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
            while step < len(self.calib_points):
                ret, frame = self.cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = fm.process(rgb)

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    pupil = self.get_pupil_center(face, w, h)
                    cv2.circle(frame, tuple(pupil), 4, (0,255,0), -1)

                    target_x, target_y, desc = self.calib_points[step]
                    cv2.putText(frame,
                                f"Look at {desc} ({step+1}/{len(self.calib_points)})",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    pupil_x_list.append(pupil[0])
                    pupil_y_list.append(pupil[1])
                    frames_collected += 1

                    if frames_collected >= self.frames_per_point:
                        avg_x = int(np.mean(pupil_x_list))
                        avg_y = int(np.mean(pupil_y_list))
                        self.calib_cam_x.append(avg_x)
                        self.calib_cam_y.append(avg_y)
                        self.calib_scr_x.append(target_x)
                        self.calib_scr_y.append(target_y)

                        step += 1
                        frames_collected = 0
                        pupil_x_list.clear()
                        pupil_y_list.clear()
                        print(f"Calibration point {step}/{len(self.calib_points)} collected")

                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                    cv2.destroyWindow("Calibration")
                    print("Calibration cancelled.")
                    return False

            # Fit linear mapping
            self.coef_x = np.polyfit(self.calib_cam_x, self.calib_scr_x, 1)
            self.coef_y = np.polyfit(self.calib_cam_y, self.calib_scr_y, 1)
            self.calibrated = True
            print("Calibration complete!")
            cv2.destroyWindow("Calibration")
            return True

    def run(self):
        print("Press 'C' to calibrate, 'M' to toggle mouse, 'Q' to quit")
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = fm.process(rgb)

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    pupil = self.get_pupil_center(face, w, h)
                    cv2.circle(frame, tuple(pupil), 4, (0,255,0), -1)

                    if self.calibrated and self.coef_x is not None and self.coef_y is not None:
                        # Map to screen
                        mx = int(np.polyval(self.coef_x, pupil[0]))
                        my = int(np.polyval(self.coef_y, pupil[1]))

                        # Apply damping
                        mx = int(self.screen_w//2 + (mx - self.screen_w//2) * self.damping)
                        my = int(self.screen_h//2 + (my - self.screen_h//2) * self.damping)

                        # Clamp to screen
                        mx = max(0, min(self.screen_w-1, mx))
                        my = max(0, min(self.screen_h-1, my))

                        # Smooth and limit jump
                        mx, my = self.smooth_coords((mx, my))
                        dx = np.clip(mx - self.prev_x, -self.max_jump, self.max_jump)
                        dy = np.clip(my - self.prev_y, -self.max_jump, self.max_jump)
                        mx = self.prev_x + dx
                        my = self.prev_y + dy
                        self.prev_x, self.prev_y = mx, my

                        if self.enable_mouse_control:
                            pyautogui.moveTo(mx, my)

                cv2.imshow("Gaze Tracking", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in [27, ord('q')]:
                    break
                elif k == ord('c'):
                    self.calibrate()
                elif k == ord('m'):
                    self.enable_mouse_control = not self.enable_mouse_control
                    print("Mouse control", "ON" if self.enable_mouse_control else "OFF")

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoothing', type=int, default=5)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--max_jump', type=int, default=30)
    args = parser.parse_args()

    tracker = GazeTracker(smoothing_window=args.smoothing,
                          frames_per_point=args.frames,
                          damping=args.damping,
                          max_jump=args.max_jump)
    tracker.run()


if __name__ == "__main__":
    main()
