import cv2
import mediapipe as mp
import numpy as np
import random
import time
 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
FUNNY_LINES = [
    "AHH HOT MAN",
    "DAYUM",
    "SHIRTLESS MODE: ON",
    "somebody call the fire dept",
    "certified hot guy moment",
    "put it back on bro",
    "Ahh SLUT"
]
 
SKIN_TRIGGER_DELTA = 0.25   # how much skin ratio must jump to count as "shirt off". Tune this.
TEXT_DURATION = 2.0         # seconds the overlay stays up
 
 
def get_torso_roi(landmarks, w, h):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    xs = [ls.x, rs.x, lh.x, rh.x]
    ys = [ls.y, rs.y, lh.y, rh.y]
    x1, x2 = int(min(xs) * w), int(max(xs) * w)
    y1, y2 = int(min(ys) * h), int(max(ys) * h)
    return x1, y1, x2, y2
 
 
def skin_ratio(roi_bgr):
    if roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask.mean() / 255.0
 
 
def main():
    cap = cv2.VideoCapture(0)
    baseline = None
    ratio = 0.0
    triggered_at = 0.0
    current_line = ""
 
    print("Press 'c' to calibrate baseline (shirt ON), 'q' to quit")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
 
        if results.pose_landmarks:
            x1, y1, x2, y2 = get_torso_roi(results.pose_landmarks.landmark, w, h)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            roi = frame[y1:y2, x1:x2]
            ratio = skin_ratio(roi)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
            if baseline is None:
                cv2.putText(frame, "Press 'c' to calibrate (shirt ON)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                delta = ratio - baseline
                cv2.putText(frame, f"skin delta: {delta:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if delta > SKIN_TRIGGER_DELTA and time.time() - triggered_at > TEXT_DURATION:
                    current_line = random.choice(FUNNY_LINES)
                    triggered_at = time.time()
 
        if current_line and time.time() - triggered_at < TEXT_DURATION:
            (tw, th), _ = cv2.getTextSize(current_line, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
            tx = (w - tw) // 2
            ty = h // 2
            cv2.rectangle(frame, (tx - 20, ty - th - 20), (tx + tw + 20, ty + 20), (0, 0, 255), -1)
            cv2.putText(frame, current_line, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
 
        cv2.imshow("Shirt-Off Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and results.pose_landmarks:
            baseline = ratio
            print(f"Calibrated baseline skin ratio: {baseline:.3f}")
        elif key == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()