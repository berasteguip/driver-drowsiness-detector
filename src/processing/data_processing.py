import cv2
from face_detection import FaceDetector
from preprocess import FacePreprocessor
import glob
import os
from tqdm import tqdm

for dataset in ('drowsy', 'active'):

    # Create ../data/processed directory if it does not exist
    os.makedirs(f"../data/processed/{dataset}", exist_ok=True)

    image_paths = glob.glob(f"../data/raw/{dataset}/*.png")
    print(f"Loading {len(image_paths)} images from dataset {dataset}...")

    face_detector = FaceDetector()
    preprocessor = FacePreprocessor()

    not_found = []

    print(f"Processing images from dataset {dataset}...")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}")
            continue

        img_name = os.path.basename(img_path)
        
        face_box = face_detector.detect(img)
        face_norm = preprocessor(img, face_box)
        
        if face_norm is not None:
            cv2.imwrite(f"../data/processed/{dataset}/{img_name}", face_norm)
            # print(f"Face found in {img_name} [OK]")  # Optional: reduce noise
        else:
            print(f"Face not found in {img_name}")
            not_found.append(img)
            
    print(f"Not found: {len(not_found)/len(image_paths)*100:.2f}% ({len(not_found)}/{len(image_paths)})")

    for i, img in enumerate(not_found):
        cv2.imshow('Face not found', img)
        print("Press any key to see next image, or Esc to close window")
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyAllWindows()
