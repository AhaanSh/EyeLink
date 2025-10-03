# pip install mediapipe opencv-python pyautogui
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh

# Left iris and eye landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]  # all 5 iris points
LEFT_EYE_CORNERS = [33, 133]  # outer, inner
LEFT_EYE_UP_DOWN = [159, 145]  # top, bottom for blink detection

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Calibration
calib_vectors = []  # pupil vectors relative to eye center
calib_scr_x = []
calib_scr_y = []
calib_points = [
    (100, 100, "top-left"),
    (screen_w-100, 100, "top-right"),
    (screen_w-100, screen_h-100, "bottom-right"),
    (100, screen_h-100, "bottom-left"),
    (screen_w//2, screen_h//2, "center")
]

step = 0
frames_collected = 0
frames_per_point = 60
calibrated = False

# Blink detection
blink_start = None
BLINK_DURATION = 5.0  # seconds

# Smoothing
smoothed_vector = np.array([0.0, 0.0])
alpha = 0.2  # smoothing factor

# Dead zone
dead_zone = 0.015  # normalized units

def get_iris_center(face_landmarks, w, h):
    points = np.array([[face_landmarks.landmark[i].x * w,
                        face_landmarks.landmark[i].y * h] for i in LEFT_IRIS])
    return points.mean(axis=0)

def get_eye_center(face_landmarks, w, h):
    coords = np.array([[face_landmarks.landmark[i].x * w,
                        face_landmarks.landmark[i].y * h] for i in LEFT_EYE_CORNERS])
    return coords.mean(axis=0)

def get_eye_width_height(face_landmarks, w, h):
    left = np.array([face_landmarks.landmark[LEFT_EYE_CORNERS[0]].x * w,
                     face_landmarks.landmark[LEFT_EYE_CORNERS[0]].y * h])
    right = np.array([face_landmarks.landmark[LEFT_EYE_CORNERS[1]].x * w,
                      face_landmarks.landmark[LEFT_EYE_CORNERS[1]].y * h])
    width = np.linalg.norm(right - left)
    top = np.array([face_landmarks.landmark[LEFT_EYE_UP_DOWN[0]].x * w,
                    face_landmarks.landmark[LEFT_EYE_UP_DOWN[0]].y * h])
    bottom = np.array([face_landmarks.landmark[LEFT_EYE_UP_DOWN[1]].x * w,
                       face_landmarks.landmark[LEFT_EYE_UP_DOWN[1]].y * h])
    height = np.linalg.norm(top - bottom)
    return width, height

def eye_aspect_ratio(face_landmarks, w, h):
    width, height = get_eye_width_height(face_landmarks, w, h)
    return height / width

def linear_map(value, src_min, src_max, dst_min, dst_max):
    # prevent division by zero
    if src_max - src_min == 0:
        return dst_min
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                pupil = get_iris_center(face, w, h)
                eye_center = get_eye_center(face, w, h)
                eye_width, eye_height = get_eye_width_height(face, w, h)

                # normalized pupil vector
                pupil_vector = (pupil - eye_center) / np.array([eye_width, eye_height])
                smoothed_vector[:] = alpha * pupil_vector + (1 - alpha) * smoothed_vector

                cv2.circle(frame, tuple(pupil.astype(int)), 4, (0,255,0), -1)

                # blink detection
                ear = eye_aspect_ratio(face, w, h)
                if ear < 0.2:
                    if blink_start is None:
                        blink_start = time.time()
                    elif time.time() - blink_start > BLINK_DURATION:
                        print("Blink held for 5 seconds. Exiting.")
                        break
                else:
                    blink_start = None

                if not calibrated:
                    target_x, target_y, desc = calib_points[step]
                    pyautogui.moveTo(target_x, target_y)
                    cv2.putText(frame,
                                f"Look at {desc} of screen ({step+1}/{len(calib_points)})",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    frames_collected += 1
                    if frames_collected >= frames_per_point:
                        calib_vectors.append(smoothed_vector.copy())
                        calib_scr_x.append(target_x)
                        calib_scr_y.append(target_y)
                        step += 1
                        frames_collected = 0
                        print(f"Calibration point {step}/{len(calib_points)} collected")

                    if step >= len(calib_points):
                        # determine min/max for x and y from calibration
                        vecs = np.array(calib_vectors)
                        min_x, max_x = vecs[:,0].min(), vecs[:,0].max()
                        min_y, max_y = vecs[:,1].min(), vecs[:,1].max()
                        calibrated = True
                        print("Calibration complete! Cursor now maps to eye movement across full screen.")
                else:
                    # dead zone near center
                    if np.linalg.norm(smoothed_vector) < dead_zone:
                        mx, my = pyautogui.position()
                    else:
                        # map using calibration min/max
                        mx = int(linear_map(smoothed_vector[0], min_x, max_x, 0, screen_w-1))
                        my = int(linear_map(smoothed_vector[1], min_y, max_y, 0, screen_h-1))
                        mx = max(0, min(screen_w-1, mx))
                        my = max(0, min(screen_h-1, my))
                        pyautogui.moveTo(mx, my)

            cv2.imshow("Eye Tracker", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
