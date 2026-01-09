from .sign_detection import detect, display_img, get_hand_landmarks
import cv2
import mediapipe as mp

class HandPassword:
    def __init__(self, password: list[str]):
        self.password = password

    def verify(self, attempt: list[str]):
        return self.password == attempt

    def start(self):

        last_sign = "NO SIGN"
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)

        try:
            with mp_hands.Hands(
                static_image_mode=False,      # Vídeo (no imágenes sueltas)
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hand:

                attempt = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)

                    hand_landmarks = get_hand_landmarks(frame, hand)

                    if hand_landmarks:
                        sign_detected = detect(frame, hand_landmarks)
                    else:
                        sign_detected = "NO SIGN"
                    if sign_detected != last_sign:
                        last_sign = sign_detected
                        print(sign_detected)   
                        if sign_detected != "NO SIGN":
                            attempt.append(sign_detected)
                            print(attempt)
                            if len(attempt) == len(self.password):
                                correct = self.verify(attempt)
                                if correct:
                                    print("Password correcto")
                                    break
                                else:
                                    print("Password incorrecto")
                                    attempt = []
                                    sign_detected = "NO SIGN"
                    cv2.putText(frame, f"Attempt: {attempt}", (frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, sign_detected, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                    key = display_img(frame, 'Webcam', max_size=800)

                    if key == ord('q') and len(attempt) > 0:
                        print("Borrando último signo")
                        attempt.pop()
                    elif key == ord('r'):
                        attempt = []
                        print("Borrando todo")

        finally:
            cv2.destroyAllWindows()
            cap.release()


if __name__ == "__main__":
    HandPassword(["ROCK", "PEACE", "SURF"]).start()