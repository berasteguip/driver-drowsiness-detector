import os
import cv2

class FaceDetector:
    '''
    Clase que detecta un único rostro en una imagen y lo dibuja
    '''
    def __init__(self):
        abs_path = os.path.join(os.path.dirname(__file__), '../../haarcascade_frontalface_default.xml')
        try:
            xml_path = os.path.relpath(abs_path, os.getcwd())
        except ValueError:
            xml_path = abs_path
        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use slightly more robust parameters (scaleFactor=1.1, minNeighbors=5)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
            
        # Nos quedamos con el más grande (el usuario principal)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return largest_face
    
    def draw(self, img):
        
        face_frame = self.detect(img)
        if face_frame is not None:
            x, y, w, h = face_frame
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
        else:
            cv2.putText(img, "Face not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def main():
    '''
    Función principal que ejecuta el detector de rostros
    '''
    webcam = cv2.VideoCapture(0)
    face_detector = FaceDetector()

    while True:
        _, img = webcam.read()
        img = cv2.flip(img, 1)

        face_detector.draw(img)
        cv2.imshow('Face detection', img)
        key = cv2.waitKey(10)

        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()