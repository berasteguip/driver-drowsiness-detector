import cv2
print("DEBUG: mesh_utils - imported cv2")
import mediapipe as mp
print("DEBUG: mesh_utils - imported mediapipe")
import numpy as np
import math

print("DEBUG: mesh_utils - initializing MP solutions...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles
print("DEBUG: mesh_utils - MP solutions initialized")

# Landmarks (MediaPipe indices)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 311, 291, 308, 78]

# ----------------- Geometric helpers -----------------
def dist(a, b):
    """2D Euclidean distance."""
    return np.linalg.norm(a - b)


def aspect_ratio(pts):
    """
    Generic aspect ratio for an array of 6 points (p0..p5)
    pts: array shape (6,2)
    """
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ----------------- EAR and MAR -----------------
def get_ear(frame, face_landmarks):
    h, w = frame.shape[:2]
    lm = face_landmarks.landmark

    left_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE])
    right_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE])

    ear_l = aspect_ratio(left_eye)
    ear_r = aspect_ratio(right_eye)
    EAR = (ear_l + ear_r) / 2.0
    return EAR

def get_mar(frame, face_landmarks):
    h, w = frame.shape[:2]
    lm = face_landmarks.landmark

    # Mouth corners (outer)
    left_corner  = np.array([lm[61].x * w, lm[61].y * h])
    right_corner = np.array([lm[291].x * w, lm[291].y * h])

    # Upper/lower lip center (inner)
    upper_lip = np.array([lm[13].x * w, lm[13].y * h])
    lower_lip = np.array([lm[14].x * w, lm[14].y * h])

    A = dist(upper_lip, lower_lip)      # vertical opening
    C = dist(left_corner, right_corner) # mouth width

    if C == 0:
        return 0.0
    return A / C


# ----------------- Head pose -----------------
def get_head_pose(frame, face_landmarks):
    h, w = frame.shape[:2]
    lm = face_landmarks.landmark

    # 2D points in pixels
    image_points = np.array([
        (lm[1].x * w,   lm[1].y * h),    # nose tip
        (lm[152].x * w, lm[152].y * h),  # chin
        (lm[33].x * w,  lm[33].y * h),   # left eye corner
        (lm[263].x * w, lm[263].y * h),  # right eye corner
        (lm[61].x * w,  lm[61].y * h),   # left mouth corner
        (lm[291].x * w, lm[291].y * h),  # right mouth corner
    ], dtype="double")

    # 3D model (relative mm)
    model_points = np.array([
        (0.0,   0.0,    0.0),     # nose tip
        (0.0,  -330.0, -65.0),    # chin
        (-225.0, 170.0, -135.0),  # left eye left corner
        (225.0, 170.0, -135.0),   # right eye right corner
        (-150.0, -150.0, -125.0), # left mouth
        (150.0, -150.0, -125.0),  # right mouth
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,           center[0]],
        [0,            focal_length, center[1]],
        [0,            0,           1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # no distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return float(pitch), float(yaw), float(roll)


# ----------------- Eye and mouth drawing -----------------
def draw_poly_norm(frame, landmarks, indices, color):
    """
    Draws a closed polyline from normalized landmark indices.
    Uses the same coordinate conversion as MediaPipe.
    """
    h, w, _ = frame.shape
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        p = mp_drawing._normalized_to_pixel_coordinates(lm.x, lm.y, w, h)
        if p is not None:
            pts.append(p)
    if len(pts) >= 2:
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
