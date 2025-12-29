from ..detection.detector import Detector
import cv2

detector = Detector()

def main():
    # Iniciamos la captura de video
    webcam = cv2.VideoCapture(0)
    # Instanciamos nuestra clase general
    mi_detector = detector

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