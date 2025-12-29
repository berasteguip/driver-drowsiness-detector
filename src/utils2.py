import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# Landmarks (MediaPipe indices)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 311, 291, 308, 78]

# ----------------- Helpers geométricos -----------------
def dist(a, b):
    """Distancia euclídea 2D."""
    return np.linalg.norm(a - b)


def aspect_ratio(pts):
    """
    Aspect ratio genérico para un arreglo de 6 puntos (p0..p5)
    pts: array shape (6,2)
    """
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ----------------- EAR y MAR -----------------
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

    # Esquinas de la boca (externas)
    left_corner  = np.array([lm[61].x * w, lm[61].y * h])
    right_corner = np.array([lm[291].x * w, lm[291].y * h])

    # Centro labio superior / inferior (internos)
    upper_lip = np.array([lm[13].x * w, lm[13].y * h])
    lower_lip = np.array([lm[14].x * w, lm[14].y * h])

    A = dist(upper_lip, lower_lip)      # apertura vertical
    C = dist(left_corner, right_corner) # ancho de la boca

    if C == 0:
        return 0.0
    return A / C


# ----------------- Head pose -----------------
def get_head_pose(frame, face_landmarks):
    h, w = frame.shape[:2]
    lm = face_landmarks.landmark

    # puntos 2D en píxeles
    image_points = np.array([
        (lm[1].x * w,   lm[1].y * h),    # nariz tip
        (lm[152].x * w, lm[152].y * h),  # barbilla
        (lm[33].x * w,  lm[33].y * h),   # ojo izq. esquina
        (lm[263].x * w, lm[263].y * h),  # ojo der. esquina
        (lm[61].x * w,  lm[61].y * h),   # boca izq.
        (lm[291].x * w, lm[291].y * h),  # boca der.
    ], dtype="double")

    # modelo 3D (mm relativos)
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

    dist_coeffs = np.zeros((4, 1))  # sin distorsión

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return float(pitch), float(yaw), float(roll)


# ----------------- Dibujo de ojos y boca -----------------
def draw_poly_norm(frame, landmarks, indices, color):
    """
    Dibuja una polilínea cerrada a partir de índices de landmarks normalizados.
    Usa la misma conversión de coordenadas que MediaPipe.
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