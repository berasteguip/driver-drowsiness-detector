from eye_detection import EyeDetector
from face_detection import FaceDetector
from mouth_detection import MouthDetector


class Detector():
    def __init__(self):
        self.eye_detector = EyeDetector()
        self.face_detector = FaceDetector()
        self.mouth_detector = MouthDetector()

    def detect(self, frame):
        face = self.face_detector.detect(frame)
        if face != None:
            eyes = self.eye_detector.detect(frame, face)
            mouth = self.mouth_detector.detect(frame, face)
            return face, eyes, mouth
        else:
            return None, None, None