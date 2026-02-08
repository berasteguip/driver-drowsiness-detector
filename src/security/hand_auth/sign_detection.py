import cv2
import mediapipe as mp
import numpy as np
import math

MARGIN = 20

def hand_rectangle(frame, hand_landmarks):

    h, w, _ = frame.shape

    xmin, xmax, ymin, ymax = hand_limits(hand_landmarks)

    # Convert to image coordinates
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
    
    # |sin(theta)| = |dy| / ||v||  (theta = angle relative to horizontal)
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

def dist_lm(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def hand_scale(pts):
    return dist_lm(pts[0], pts[5]) + 1e-9

def rect_aspect_ratio_from_landmarks(hand_landmarks):
    # Height / width ratio of the bounding box in normalized coordinates
    xmin, xmax, ymin, ymax = hand_limits(hand_landmarks)
    width = (xmax.x - xmin.x) + 1e-9
    height = (ymax.y - ymin.y) + 1e-9
    return height / width

#####################
# Hand signs
#####################

def detect_surf(hand_landmarks, length_threshold = 0.7, max_sin_angle = 0.5, verbose = False):
    """
    Conditions:
    1) Thumb and pinky very far apart (ratio over diagonal)
    2) Thumb-pinky segment roughly horizontal (small sin(theta))
    3) The two highest tips (smallest y) must be thumb and pinky (any order)
    """

    # (3) Check the "highest" tips
    tips = list(get_tips(hand_landmarks).items())
    tips_sorted = sorted(tips, key=lambda x: x[1].y)  # smaller y = higher
    top2 = {tips_sorted[0][0], tips_sorted[1][0]}
    if top2 != {"p", "m"}:
        return False

    # (1) + (2) Original checks
    th_pk_dist, diag = finger_dist(hand_landmarks, 'p', 'm')
    diag_ratio = th_pk_dist / (diag + 1e-9)

    fingers = get_fingers(hand_landmarks)
    thumb_tip, pinky_tip = fingers['p'][-1], fingers['m'][-1]
    ang = horizontality(thumb_tip, pinky_tip)

    if verbose:
        print('Thumb-pinky distance', th_pk_dist)
        print('Frame diagonal', diag)
        print('Ratio', diag_ratio)
        print('sin(theta)', ang)

    return (diag_ratio > length_threshold) and (ang < max_sin_angle)

def detect_rock(hand_landmarks):
    
    i_m_dist, diag = finger_dist(hand_landmarks, 'i', 'm')
    
    tips = list(get_tips(hand_landmarks).items())

    tips_sorted = sorted(tips, key=lambda x: x[1].y)

    if tips_sorted[0][0] in ["i", "m"]:
        if tips_sorted[1][0] in ["i", "m"]:
            return True
    return False

def detect_peace(hand_landmarks, min_vertical_ratio=2.0):
    """
    In addition to your logic:
    - The hand bounding box must be clearly vertical:
      height/width >= 2.0  (equivalent to at least 1:2 in width:height)
    """

    # Vertical rectangle check (height/width >= 2)
    aspect = rect_aspect_ratio_from_landmarks(hand_landmarks)
    if aspect < min_vertical_ratio:
        return False

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

# Alien hand (Spock)
def detect_vulcan(hand_landmarks,
                  pair_thr=0.30,
                  gap_thr=0.55,
                  ratio_k=1.8):
    pts = hand_landmarks.landmark
    S = hand_scale(pts)

    # TIPs
    d_im = dist_lm(pts[8],  pts[12]) / S
    d_rp = dist_lm(pts[16], pts[20]) / S
    d_mr = dist_lm(pts[12], pts[16]) / S

    tip_ok = (d_im < pair_thr) and (d_rp < pair_thr) and (d_mr > gap_thr) and (d_mr > ratio_k * max(d_im, d_rp))

    # PIPs (backup)
    d_im_p = dist_lm(pts[6],  pts[10]) / S
    d_rp_p = dist_lm(pts[14], pts[18]) / S
    d_mr_p = dist_lm(pts[10], pts[14]) / S

    pip_ok = (d_im_p < pair_thr) and (d_rp_p < pair_thr) and (d_mr_p > gap_thr) and (d_mr_p > ratio_k * max(d_im_p, d_rp_p))

    return tip_ok or pip_ok

def detect(frame, hand_landmarks):
    
    if hand_landmarks == None:
        return 'NO SIGN'

    rect = hand_rectangle(frame, hand_landmarks)

    sign_detected = 'NO SIGN'

    if rect:
        x1, y1, x2, y2 = rect

        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 4)
        crop = frame[y1:y2, x1:x2]
        display_img(crop, 'Hand', max_size=500)

        vulcan = detect_vulcan(hand_landmarks)
        if vulcan:
            sign_detected = 'VULCAN'

        surf_ok = detect_surf(hand_landmarks)
        if surf_ok:
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
        static_image_mode=False,      # Video (not still images)
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
