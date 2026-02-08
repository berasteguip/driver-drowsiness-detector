from .eye_detection import EyeDetector
from .face_detection import FaceDetector
from .mouth_detection import MouthDetector
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
        # 1. Run detection
        face, eyes, mouth = self.detect(frame)

        # 2. Draw Face (Blue)
        if face is not None:
            xf, yf, wf, hf = face
            cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (xf, yf - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 3. Draw Eyes (Green)
        if eyes is not None:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eyes not detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. Draw Mouth (Yellow)
        if mouth is not None:
            mx, my, mw, mh = mouth
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Mouth not detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

def main():
    # Start video capture
    webcam = cv2.VideoCapture(0)
    # Instantiate our general detector class
    my_detector = Detector()

    print("Press ESC to exit...")

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # Mirror for a more natural feel
        frame = cv2.flip(frame, 1)

        # Call draw, which now detects and draws everything
        processed_frame = my_detector.draw(frame)

        # Show result
        cv2.imshow('Full Detection - Face', processed_frame)

        # Exit with ESC (27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
