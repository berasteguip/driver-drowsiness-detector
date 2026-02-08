import os
import cv2
from config import *

class MouthDetector:
    '''
    Class that detects the mouth in an image and returns global coordinates
    '''
    def __init__(self):
        # Load the mouth classifier
        abs_path = MOUTH_CASCADE_PATH
        try:
            xml_path = os.path.relpath(abs_path, os.getcwd())
        except ValueError:
            xml_path = abs_path
        self.mouth_cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, img, face_frame):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Extract face coordinates: face_frame = (x_face, y_face, w_face, h_face)
        xf, yf, wf, hf = face_frame
        gray_face = gray[yf :yf+hf, xf:xf+wf]
        
        # 3. Detect mouths in the face region
        mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        # If no mouth is detected, return None
        if len(mouths) == 0:
            return None
            
        # 4. Sort by area and take the largest
        sorted_mouths = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)
        mx, my, mw, mh = sorted_mouths[0] # Coordinates relative to face
        
        # 5. ADJUSTMENT: Add face offset to get global coordinates
        # This allows the mouth to match the original image
        global_mouth = (xf + mx, yf + my, mw, mh)
        
        return global_mouth
