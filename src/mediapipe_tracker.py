from utils2 import *
import cv2

# ----------------- Main -----------------
def mediapipe_tracker():
    cap = cv2.VideoCapture(0)

    # Umbrales
    EAR_THRESH = 0.21
    EAR_CONSEC_FRAMES = 3
    MAR_THRESH = 0.35

    counter_closed = 0
    total_blinks = 0
    perclos_window_sec = 60
    perclos_history = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # se ve en modo espejo
            if not ret:
                break

            # Si quieres espejo, hazlo aquí:
            # frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            estado = "NO FACE"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                # Métricas
                EAR = get_ear(frame, landmarks)
                MAR = get_mar(frame, landmarks)

                pitch, yaw, roll = get_head_pose(frame, landmarks)

                # Face mesh completo
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

                # Ojos y boca encima del mesh
                draw_poly_norm(frame, landmarks, LEFT_EYE,  (0, 255, 0))
                draw_poly_norm(frame, landmarks, RIGHT_EYE, (0, 255, 0))
                draw_poly_norm(frame, landmarks, MOUTH,     (0,   0, 255))

                # Lógica de parpadeo y PERCLOS
                if EAR < EAR_THRESH:
                    counter_closed += 1
                    is_closed = True
                else:
                    if counter_closed >= EAR_CONSEC_FRAMES:
                        total_blinks += 1
                    counter_closed = 0
                    is_closed = False

                ts = cv2.getTickCount() / cv2.getTickFrequency()
                perclos_history.append((ts, is_closed))
                perclos_history = [(t, c) for (t, c) in perclos_history
                                   if ts - t <= perclos_window_sec]
                if perclos_history:
                    closed_count = sum(1 for _, c in perclos_history if c)
                    perclos = closed_count / len(perclos_history)
                else:
                    perclos = 0.0

                yawning = MAR > MAR_THRESH

                estado = f"EAR:{EAR:.3f} MAR:{MAR:.3f} PERCLOS:{perclos:.2f} PITCH:{pitch:.1f}"
                if yawning:
                    estado += " BOSTEZO"

            cv2.putText(frame, estado, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Fatigue metrics", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()