import cv2
from face_detection import FaceDetector

class MouthDetector:
    '''
    Clase que detecta la boca en una imagen y lo dibuja
    '''
    def __init__(self):
        self.mouth_cascade = cv2.CascadeClassifier('../haarcascade_mcs_mouth.xml')
        self.face_detector = FaceDetector()

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use slightly more robust parameters (scaleFactor=1.1, minNeighbors=5)
        face_frame = self.face_detector.detect(img)
        if face_frame is None:
            return None
        mouths = self.mouth_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(mouths) == 0:
            return None
            
        # Nos quedamos con el más pequeño (el usuario principal)
        lowest_mouth = min(mouths, key=lambda m: m[1])
        return lowest_mouth
    
    def draw(self, img):
        
        mouth_frame = self.detect(img)
        if mouth_frame is not None:
            x, y, w, h = mouth_frame
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
        else:
            cv2.putText(img, "Mouth not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def main():
    '''
    Función principal que ejecuta el detector de bocas
    '''
    webcam = cv2.VideoCapture(0)
    mouth_detector = MouthDetector()

    while True:
        _, img = webcam.read()
        img = cv2.flip(img, 1)

        mouth_detector.draw(img)
        cv2.imshow('Mouth detection', img)
        key = cv2.waitKey(10)

        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()