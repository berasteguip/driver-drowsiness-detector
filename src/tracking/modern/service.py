print("DEBUG: modern/service.py - start")
try:
    from .mesh_utils import *
    print("DEBUG: modern/service.py - imported mesh_utils")
except ImportError:
    # Fallback in case this is run as a standalone script for testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from mesh_utils import *

import cv2
import mediapipe as mp
import time  # <-- Added
print("DEBUG: modern/service.py - imports done")

# ----------------- Main -----------------
def run_modern_tracker():
    cap = cv2.VideoCapture(0)

    # --- THRESHOLD CONFIGURATION ---
    EAR_THRESH = 0.21          # Eye opening threshold to consider eye closed
    EAR_CONSEC_FRAMES = 3      # Frames to validate a blink
    DROWSY_CONSEC_FRAMES = 30  # Consecutive closed frames for DROWSINESS alert (approx 1s)
    MAR_THRESH = 0.50          # Mouth threshold for yawn

    # State variables
    counter_closed = 0
    total_blinks = 0
    perclos_window_sec = 60
    perclos_history = []
    
    # Driver status
    driver_status = "ACTIVE"
    color_status = (0, 255, 0) # Green by default

    # FPS variables
    prev_time = 0.0
    fps = 0.0

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_styles = mp.solutions.drawing_styles

    print("Starting Face Mesh...")
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror effect for a natural feel
            frame = cv2.flip(frame, 1)

            # --- FPS ---
            cur_t = time.time()
            if prev_time > 0:
                dt = cur_t - prev_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps = (0.9 * fps + 0.1 * inst_fps) if fps > 0 else inst_fps
            prev_time = cur_t

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Reset default status if there is no face
            driver_status = "NO FACE"
            color_status = (100, 100, 100) # Gray

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                # 1. Compute metrics
                EAR = get_ear(frame, landmarks)
                MAR = get_mar(frame, landmarks)
                pitch, yaw, roll = get_head_pose(frame, landmarks)

                # 2. Draw Mesh (technical visualization)
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw key contours
                draw_poly_norm(frame, landmarks, LEFT_EYE,  (0, 255, 0))
                draw_poly_norm(frame, landmarks, RIGHT_EYE, (0, 255, 0))
                draw_poly_norm(frame, landmarks, MOUTH,     (0, 0, 255))

                # 3. Drowsiness detection logic
                is_closed = EAR < EAR_THRESH
                is_yawning = MAR > MAR_THRESH

                if is_closed:
                    counter_closed += 1
                    # If eyes are closed for many consecutive frames -> Drowsy
                    if counter_closed >= DROWSY_CONSEC_FRAMES:
                        driver_status = "DROWSY"
                        color_status = (0, 0, 255) # Red
                else:
                    # If eyes open, check if it was a blink
                    if counter_closed >= EAR_CONSEC_FRAMES:
                        total_blinks += 1
                    counter_closed = 0
                    
                    # If yawning -> Warning
                    if is_yawning:
                        driver_status = "YAWNING"
                        color_status = (0, 165, 255) # Orange
                    else:
                        driver_status = "ACTIVE"
                        color_status = (0, 255, 0) # Green

                # 4. PERCLOS calculation (history)
                ts = cv2.getTickCount() / cv2.getTickFrequency()
                perclos_history.append((ts, is_closed))
                # Keep only the last 60 seconds
                perclos_history = [(t, c) for (t, c) in perclos_history if ts - t <= perclos_window_sec]
                
                if perclos_history:
                    closed_count = sum(1 for _, c in perclos_history if c)
                    perclos = closed_count / len(perclos_history)
                else:
                    perclos = 0.0

                # If PERCLOS is high (> 0.30), force Drowsy state too
                if perclos > 0.30 and driver_status != "NO FACE":
                    driver_status = "DROWSY (PERCLOS)"
                    color_status = (0, 0, 255)

                # 5. Data visualization
                info_text = f"EAR: {EAR:.2f} | MAR: {MAR:.2f} | PERCLOS: {perclos:.2f}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # LARGE STATUS INDICATOR
                # Draw a background rectangle for readability
                cv2.rectangle(frame, (10, 50), (300, 100), (0, 0, 0), -1)
                cv2.putText(frame, driver_status, (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_status, 3)

            # --- Draw FPS in the top-right corner ---
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x = frame.shape[1] - tw - 10
            y = 10 + th
            cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Fatigue Monitor (MediaPipe)", frame)
            
            # Exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_modern_tracker()
