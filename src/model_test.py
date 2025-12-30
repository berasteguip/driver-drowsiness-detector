from __future__ import annotations

import cv2
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to ensure we can import from src
# Assuming this script is run from src/ or project root, we need to locate 'src'
# The user seems to run from project root usually, but let's be robust.
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.detection.face_detection import FaceDetector
from src.detection.eye_detection import EyeDetector
from src.processing.eyes_features import HOGFeatureExtractor, HOGConfig

def load_models(models_dir: Path):
    """Load the left and right eye XGBoost models."""
    left_model_path = models_dir / "xgb_eye_left.pkl"
    right_model_path = models_dir / "xgb_eye_right.pkl"

    if not left_model_path.exists() or not right_model_path.exists():
        raise FileNotFoundError(f"Models not found in {models_dir}. Please train them first.")

    model_left = joblib.load(left_model_path)
    model_right = joblib.load(right_model_path)
    return model_left, model_right

def preprocess_eye(img: np.ndarray, eye_rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """
    Extracts eye region, converts to grayscale, and resizes to 32x32.
    eye_rect is (x, y, w, h)
    """
    x, y, w, h = eye_rect
    
    # Boundary checks
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > img.shape[1]: w = img.shape[1] - x
    if y + h > img.shape[0]: h = img.shape[0] - y

    eye_img = img[y:y+h, x:x+w]
    if eye_img.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 32x32 as used in training
    try:
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        return resized
    except cv2.error:
        return None

def main():
    # Setup paths
    # Expect models in <project_root>/models
    models_dir = project_root / "models"
    
    print("Loading models...")
    try:
        model_left, model_right = load_models(models_dir)
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize detectors and extractors
    face_detector = FaceDetector()
    eye_detector = EyeDetector()
    
    # Same config as training
    hog_cfg = HOGConfig(
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm="L2-Hys"
    )
    feature_extractor = HOGFeatureExtractor(hog_cfg)

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Controls ---")
    print("CENTER ENTER: Capture and Evaluate")
    print("ESC: Quit")
    print("----------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Draw Face and Eyes for live feedback (optional, but good for user)
        # Note: We do detection every frame for visualization, but only predict on Enter
        # To make it faster, we could skip detection on every frame if it's slow, 
        # but modern CPUs handle robust cascades okay-ish.
        # Let's keep it simple: Real-time visualization helps positioning.
        
        display_frame = frame.copy()
        face_frame = face_detector.detect(frame) # Wait, verify API
        
        # Checking existing API:
        # FaceDetector.detect(img) returns rect (x,y,w,h) or None
        # FaceDetector.draw(img) draws it
        
        # EyeDetector.detect(img, face_frame) returns (left_eye_rect, right_eye_rect)
        # EyeDetector.draw(img, eyes_frame)
        
        face_detector.draw(display_frame) # This re-detects internally in the existing class structure... 
        # Actually FaceDetector.draw calls detect internally. 
        # To avoid double detection, we might need to modify or just use detect directly.
        
        # Optimization: Call detect once.
        face_rect = face_detector.detect(frame)
        
        if face_rect is not None:
            # Draw face manualy since we have the rect
            fx, fy, fw, fh = face_rect
            cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            
            eyes_rects = eye_detector.detect(frame, face_rect)
            if eyes_rects is not None:
                eye_detector.draw(display_frame, eyes_rects)

        cv2.imshow("Driver Drowsiness Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == 13: # ENTER
            print("\nEvaluating frame...")
            if face_rect is None:
                print(">> No face detected. Try again.")
                continue

            if eyes_rects is None:
                print(">> Face found, but eyes NOT detected. Try again.")
                continue

            # Process Eyes
            # eyes_rects is (left_eye, right_eye) in existing API
            left_rect, right_rect = eyes_rects
            
            # Preprocess
            left_img = preprocess_eye(frame, left_rect)
            right_img = preprocess_eye(frame, right_rect)

            if left_img is None or right_img is None:
                print(">> Error processing eye images (resize failed?).")
                continue

            # Feature Extraction
            feat_left = feature_extractor.extract(left_img)
            feat_right = feature_extractor.extract(right_img)

            # Predict (Probabilities)
            # classes are 0: active, 1: drowsy usually? 
            # In train.py: active=0, drowsy=1 is inferred from folder names usually?
            # Let's check train.py loading:
            # a = np.load(.../active/...) y=0? 
            # In eyes_features.py: class_to_label = {"active": 0, "drowsy": 1}
            # So 0 is ACTIVE, 1 is DROWSY.
            
            # predict_proba returns [prob_class_0, prob_class_1]
            # We want probability of Drowsiness (Class 1)
            prob_left = model_left.predict_proba(feat_left.reshape(1, -1))[0][1]
            prob_right = model_right.predict_proba(feat_right.reshape(1, -1))[0][1]

            avg_prob = (prob_left + prob_right) / 2.0
            
            status = "DROWSY" if avg_prob > 0.5 else "ACTIVE"
            conf = avg_prob if status == "DROWSY" else (1 - avg_prob)

            print(f">> Left Eye Drowsy Prob : {prob_left:.4f}")
            print(f">> Right Eye Drowsy Prob: {prob_right:.4f}")
            print(f">> Average Drowsy Prob  : {avg_prob:.4f}")
            print(f">> RESULT: {status} (Confidence: {conf:.2%})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
