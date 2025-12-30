import cv2
import mediapipe as mp
import numpy as np
import math

MARGIN = 20

def hand_rectangle(frame, hand_landmarks):

    h, w, _ = frame.shape

    xmin, xmax, ymin, ymax = hand_limits(hand_landmarks)

    # Pasar a coordenadas de imagen
    x1 = int(xmin.x * w) - MARGIN
    x2 = int(xmax.x * w) + MARGIN
    y1 = int(ymin.y * h) - MARGIN
    y2 = int(ymax.y * h) + MARGIN

    return (x1, y1, x2, y2)  # x1,y1,x2,y2

def get_hand_landmarks(frame, hand):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_landmarks = hand.process(image_rgb).multi_hand_landmarks
    
    if type(hand_landmarks) == list:
        return hand_landmarks[0]
    return hand_landmarks

def get_fingers(hand_landmarks):

    pts = hand_landmarks.landmark

    fingers = {
                        "pl": [pts[0]],
                        "p": pts[1:5],
                        "i": pts[5:9],
                        "c": pts[9:13],
                        "a": pts[13:17],
                        "m": pts[17:21],
                    }
    return fingers

def hand_limits(hand_landmarks) -> tuple:

    pts = hand_landmarks.landmark
    xmin = min(pts, key=lambda p: p.x)
    xmax = max(pts, key=lambda p: p.x)
    ymin = min(pts, key=lambda p: p.y)
    ymax = max(pts, key=lambda p: p.y)

    return (xmin, xmax, ymin, ymax)

def hand_diagonal(hand_landmarks):
            
    xmin, xmax, ymin, ymax = hand_limits(hand_landmarks)

    dx = xmax.x - xmin.x
    dy = ymax.y - ymin.y

    return math.hypot(dx, dy)

def finger_dist(hand_landmarks, f_1: str, f_2: str):
    
    fingers = get_fingers(hand_landmarks)
    
    tip_1, tip_2 = fingers[f_1][-1], fingers[f_2][-1]     # tips of each finger

    dist = np.sqrt((tip_1.x - tip_2.x)**2 + (tip_1.y - tip_2.y) ** 2)

    diag = hand_diagonal(hand_landmarks)

    return dist, diag

def display_img(frame, imgname='img', max_size=300):

    if frame.size == 0:
        return

    h, w, _ = frame.shape

    if h > w:
        resized = cv2.resize(frame, (w*max_size // h, max_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(frame, (max_size, h*max_size // w), interpolation=cv2.INTER_LINEAR)

    cv2.imshow(imgname, resized)
    return cv2.waitKey(1)

def horizontality(tip1, tip2):

    dx = tip1.x - tip2.x
    dy = tip1.y - tip2.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        return
    
    # |sin(theta)| = |dy| / ||v||  (theta = ángulo respecto a horizontal)
    sin_theta = abs(dy) / norm
    return sin_theta

def get_tips(hand_landmarks):

    fingers= get_fingers(hand_landmarks)
    tips = {
        'p': fingers['p'][-1],
        'i': fingers['i'][-1],
        'c': fingers['c'][-1],
        'a': fingers['a'][-1],
        'm': fingers['m'][-1],
    }
    return tips
    

def detect_surf(hand_landmarks, length_threshold = 0.7, max_sin_angle = 0.5, verbose = False):

    th_pk_dist, diag = finger_dist(hand_landmarks, 'p', 'm')
    
    diag_ratio = th_pk_dist / diag
    
    fingers = get_fingers(hand_landmarks)
    thumb_tip, pinky_tip = fingers['p'][-1], fingers['m'][-1]

    ang = horizontality(thumb_tip, pinky_tip)

    if verbose:
        print('Distancia pulgar meñique', th_pk_dist)
        print('Diagonal marco', diag)
        print('Ratio', diag_ratio)

    # print('angulo', ang)

    return (diag_ratio > length_threshold) and (ang < max_sin_angle)

def detect_rock(hand_landmarks):
    
    i_m_dist, diag = finger_dist(hand_landmarks, 'i', 'm')
    
    tips = list(get_tips(hand_landmarks).items())

    tips_sorted = sorted(tips, key=lambda x: x[1].y)

    if tips_sorted[0][0] in ["i", "m"]:
        if tips_sorted[1][0] in ["i", "m"]:
            return True
    return False

def detect_peace(hand_landmarks):
    
    i_c_dist, diag = finger_dist(hand_landmarks, 'i', 'c')
    
    tips = list(get_tips(hand_landmarks).items())

    tips_sorted = sorted(tips, key=lambda x: x[1].y)

    max_tips = tips_sorted[:2]
    other_tips = tips_sorted[2:]

    for tip in max_tips:
        if tip[0] not in ["i", "c"]:
            return False

    minmax_tip = max_tips[1]
    maxmin_tip = other_tips[0]

    dst, diag = finger_dist(hand_landmarks, minmax_tip[0], maxmin_tip[0])

    return dst / diag > 0.3

def detect(frame, hand_landmarks):
    rect = hand_rectangle(frame, hand_landmarks)

    sign_detected = 'NO SIGN'

    if rect:
        x1, y1, x2, y2 = rect

        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 4)
        crop = frame[y1:y2, x1:x2]
        display_img(crop, 'Hand', max_size=500)

        surf_ratio = detect_surf(hand_landmarks)
        if surf_ratio > 0.7:
            sign_detected = 'SURF'

        rock = detect_rock(hand_landmarks)
        if rock:
            sign_detected = 'ROCK'

        peace = detect_peace(hand_landmarks)
        if peace:
            sign_detected = 'PEACE'
    return sign_detected

if __name__ == "__main__":

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,      # Vídeo (no imágenes sueltas)
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hand:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            hand_landmarks = get_hand_landmarks(frame, hand)

            sign_detected = detect(frame, hand_landmarks)

            cv2.putText(frame, sign_detected, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            
            display_img(frame, 'Webcam', max_size=800)
            """key = cv2.waitKey()

            if key == ord('q'):
                break"""