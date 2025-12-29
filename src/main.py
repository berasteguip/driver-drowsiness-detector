import cv2
from face_detection import FaceDetector
from preprocess import FacePreprocessor
from utils import *


def main():
    
    webcam = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    preprocessor = FacePreprocessor()

    while True:
        _, img = webcam.read()
        img = cv2.flip(img, 1)

        face_box = face_detector.detect(img)
        face_norm = preprocessor(img, face_box)
        draw(img, face_box)
        if face_norm is not None:
            cv2.imshow("Face norm", face_norm)
        cv2.imshow("Face detection", img)
        key = cv2.waitKey(10)

        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
