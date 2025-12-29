import cv2
import mediapipe as mp

# Módulos de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abrir webcam
cap = cv2.VideoCapture(0)

FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}

with mp_hands.Hands(
    static_image_mode=False,      # Vídeo (no imágenes sueltas)
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape

        # OpenCV usa BGR, MediaPipe espera RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar frame
        results = hands.process(image_rgb)

        # Dibujar landmarks si hay manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            fingertip_ids = {
                        "TH": mp_hands.HandLandmark.THUMB_TIP,
                        "IN": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        "MI": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        "AN": mp_hands.HandLandmark.RING_FINGER_TIP,
                        "ME": mp_hands.HandLandmark.PINKY_TIP,
                    }

            for label, lm_id in fingertip_ids.items():
                lm = hand_landmarks.landmark[lm_id]
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Círculo en la yema
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                # Etiqueta (abreviatura del dedo)
                cv2.putText(frame, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar resultado
        cv2.imshow("MediaPipe Hands", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
