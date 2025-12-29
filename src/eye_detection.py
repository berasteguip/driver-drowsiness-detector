import cv2
from face_detection import FaceDetector

class EyeDetector:
    '''
    Clase que detecta un par de ojos en una imagen y devuelve coordenadas globales
    '''
    def __init__(self):
        # Mantenemos tu ruta original
        self.eye_cascade = cv2.CascadeClassifier('../haarcascade_eye.xml')

    def detect(self, img, face_frame):
        # 1. Convertimos a gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Extraemos coordenadas de la cara
        xf, yf, wf, hf = face_frame
        gray_face = gray[yf:yf+hf, xf:xf+wf]
        
        # 3. Detectamos los ojos en la región de la cara
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        # Si no detecta al menos dos, devolvemos None
        if len(eyes) < 2:
            return None
            
        # 4. Ordenamos por área (más pequeños primero como tenías antes)
        sorted_eyes = sorted(eyes, key=lambda f: f[2] * f[3], reverse=False)
        
        # Ojo 1: Extraemos y sumamos posición de la cara
        ex1, ey1, ew1, eh1 = sorted_eyes[0]
        eye1_global = (xf + ex1, yf + ey1, ew1, eh1)
        
        # Ojo 2: Extraemos y sumamos posición de la cara
        ex2, ey2, ew2, eh2 = sorted_eyes[1]
        eye2_global = (xf + ex2, yf + ey2, ew2, eh2)
        
        return (eye1_global, eye2_global)
    
    def draw(self, img, eyes_frame):
        # Si recibimos la tupla de dos ojos, los dibujamos
        if eyes_frame is not None:
            for (ex, ey, ew, eh) in eyes_frame:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            cv2.putText(img, "Eyes not found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def main():
    webcam = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    eye_detector = EyeDetector()

    while True:
        _, img = webcam.read()
        img = cv2.flip(img, 1)

        # 1. Primero necesitamos el frame de la cara
        face_frame = face_detector.detect(img)
        
        if face_frame is not None:
            # 2. Detectamos ojos pasando el face_frame
            eyes_frame = eye_detector.detect(img, face_frame)
            # 3. Dibujamos
            eye_detector.draw(img, eyes_frame)
        else:
            cv2.putText(img, "Face not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Eye detection', img)
        if cv2.waitKey(10) == 27: # ESC para salir
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()