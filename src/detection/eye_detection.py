import os
import cv2
from .face_detection import FaceDetector
from config import *

class EyeDetector:
    '''
    Class that detects a pair of eyes in an image and returns global coordinates
    '''
    def __init__(self):
        # Use absolute path relative to the file, then try to convert to CWD-relative
        # to avoid encoding issues with special characters in Windows/OpenCV paths
        abs_path = EYE_CASCADE_PATH
        try:
            # If we're at the project root, this returns "haarcascade_eye.xml" which is safe
            xml_path = os.path.relpath(abs_path, os.getcwd())
        except ValueError:
            xml_path = abs_path
            
        self.eye_cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, img, face_frame):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Extract face coordinates
        xf, yf, wf, hf = face_frame
        gray_face = gray[yf:yf+hf, xf:xf+wf]
        
        # 3. Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        # If fewer than two eyes, return None
        if len(eyes) < 2:
            return None
            
        # 4. Sort by area (smaller first as before)
        sorted_eyes = sorted(eyes, key=lambda f: f[2] * f[3], reverse=False)
        
        sorted_eyes = sorted_eyes[:2]
        sorted_eyes = sorted(sorted_eyes, key=lambda f: f[0], reverse=False)

        # Eye 1: Extract and add face offset
        ex1, ey1, ew1, eh1 = sorted_eyes[0]
        left_eye = (xf + ex1, yf + ey1, ew1, eh1)
        
        # Eye 2: Extract and add face offset
        ex2, ey2, ew2, eh2 = sorted_eyes[1]
        right_eye = (xf + ex2, yf + ey2, ew2, eh2)
        
        return (left_eye, right_eye)
    
    def draw(self, img, eyes_frame):
        # If we receive the tuple of two eyes, draw them
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

        # 1. First we need the face frame
        face_frame = face_detector.detect(img)
        
        if face_frame is not None:
            # 2. Detect eyes by passing the face_frame
            eyes_frame = eye_detector.detect(img, face_frame)
            # 3. Draw
            eye_detector.draw(img, eyes_frame)
        else:
            cv2.putText(img, "Face not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Eye detection', img)
        if cv2.waitKey(10) == 27: # ESC to exit
            break

    webcam.release()
    cv2.destroyAllWindows()
