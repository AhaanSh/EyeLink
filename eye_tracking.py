import cv2 
import numpy as np 
import dlib 

cap = cv2.VideoCapture(0) 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_region(landmarks, eye_points):
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points], dtype=np.int32)

last_pupil = None  # store last known pupil center

while True: 
    _, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grey) 

    for face in faces: 
        landmarks = predictor(grey, face)

        # Right eye landmarks
        right_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_region = get_eye_region(landmarks, right_eye_points)

        # Create mask
        mask = np.zeros_like(grey)
        cv2.fillPoly(mask, [right_eye_region], 255)
        eye = cv2.bitwise_and(grey, grey, mask=mask)

        # Crop eye region
        min_x, max_x = np.min(right_eye_region[:,0]), np.max(right_eye_region[:,0])
        min_y, max_y = np.min(right_eye_region[:,1]), np.max(right_eye_region[:,1])
        gray_eye = eye[min_y:max_y, min_x:max_x]

        # Adaptive threshold
        threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 5)

        # Clean noise
        kernel = np.ones((3,3), np.uint8)
        threshold_eye = cv2.morphologyEx(threshold_eye, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pupil = (min_x+cx, min_y+cy)
                last_pupil = pupil
        else:
            pupil = last_pupil  # fallback if detection fails

        if last_pupil:
            cv2.circle(frame, last_pupil, 3, (0, 0, 255), -1)

        # Debug windows
        cv2.imshow("Threshold Eye", threshold_eye)

    cv2.imshow("Frame", frame) 
    key = cv2.waitKey(1)
    if key == 27: 
        break 

cap.release()
cv2.destroyAllWindows()
