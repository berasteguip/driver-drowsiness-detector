from eye_detection import EyeDetector
from face_detection import FaceDetector
from mouth_detection import MouthDetector
import cv2

class Detector():
    def __init__(self):
        self.eye_detector = EyeDetector()
        self.face_detector = FaceDetector()
        self.mouth_detector = MouthDetector()

    def detect(self, frame):
        face = self.face_detector.detect(frame)
        if face is not None:
            eyes = self.eye_detector.detect(frame, face)
            mouth = self.mouth_detector.detect(frame, face)
            return face, eyes, mouth
        else:
            return None, None, None

    def draw(self, frame):
        # 1. Ejecutamos la detección
        face, eyes, mouth = self.detect(frame)

        # 2. Dibujamos la Cara (Azul)
        if face is not None:
            xf, yf, wf, hf = face
            cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
            cv2.putText(frame, "Cara", (xf, yf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Cara no detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 3. Dibujamos los Ojos (Verde)
        if eyes is not None:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Ojos no detectados", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. Dibujamos la Boca (Amarillo)
        if mouth is not None:
            mx, my, mw, mh = mouth
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Boca no detectada", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

def main():
    # Iniciamos la captura de video
    webcam = cv2.VideoCapture(0)
    # Instanciamos nuestra clase general
    mi_detector = Detector()

    print("Presiona ESC para salir...")

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # Espejo para que sea más natural
        frame = cv2.flip(frame, 1)

        # Llamamos al método draw que ahora detecta y dibuja todo
        frame_procesado = mi_detector.draw(frame)

        # Mostramos el resultado
        cv2.imshow('Deteccion Completa - Facial', frame_procesado)

        # Salir con la tecla ESC (27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()