import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from ..detection.eye_detection import EyeDetector
from .preprocess import EyePreprocessor

def main():
    # Initialize the eye detector
    eye_detector = EyeDetector()
    
    # Initialize the eye preprocessor (32x32 for HOG, no extra CLAHE because it's already processed)
    eye_preprocessor = EyePreprocessor(output_size=32, margin=0.1, use_clahe=False)

    # Define paths
    # project_root/
    #   driver-drowsiness-detector/ (BASE_DIR)
    #     src/
    #   data/ (Sibling)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'face\processed')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'eyes')
    LEFT_DIR = os.path.join(OUTPUT_DIR, 'active/left')
    RIGHT_DIR = os.path.join(OUTPUT_DIR, 'active/right')
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    LEFT_DIR = os.path.join(OUTPUT_DIR, 'drowsy/left')
    RIGHT_DIR = os.path.join(OUTPUT_DIR, 'drowsy/right')

    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    datasets = ['active', 'drowsy']

    for dataset in datasets:
        input_path = os.path.join(PROCESSED_DIR, dataset)
        output_path = os.path.join(OUTPUT_DIR, dataset)
        
        # Create output directory if it does not exist
        os.makedirs(output_path, exist_ok=True)

        # Find images (png or jpg)
        image_files = glob.glob(os.path.join(input_path, "*.png")) + \
                      glob.glob(os.path.join(input_path, "*.jpg"))
        
        print(f"Processing {len(image_files)} images from '{dataset}' in '{input_path}'...")
        
        processed_count = 0
        eyes_extracted = 0

        for img_file in tqdm(image_files):
            # Use numpy + imdecode to support Windows paths with accents
            # cv2.imread silently fails with non-ASCII characters in the path
            try:
                stream = np.fromfile(img_file, dtype=np.uint8)
                img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            except Exception:
                print(f"Error loading image: {img_file}")
                img = None

            if img is None:
                continue
            
            # Since the image IS THE FACE already (processed), the face frame is the whole image
            h, w = img.shape[:2]
            face_frame = (0, 0, w, h)

            # Detect eyes
            eyes_coords = eye_detector.detect(img, face_frame)
            

            if eyes_coords is not None:
                
                # Extract each eye using the preprocessor
                for i, eye_box in enumerate(eyes_coords):
                    
                    # Use the preprocessor to extract and normalize (32x32)
                    eye_img = eye_preprocessor(img, eye_box)
                    
                    if eye_img is not None:
                        # Save eye
                        # Save eye using imencode and tofile to support paths with accents
                        
                        side = 'left' if i == 0 else 'right'

                        base_name = os.path.basename(img_file)
                        name_parts = os.path.splitext(base_name)
                        save_name = f"{name_parts[0]}_eye{i+1}.png"
                        side_path = os.path.join(output_path, side)
                        save_path = os.path.join(side_path, save_name)
                        
                        success, buffer = cv2.imencode(".png", eye_img)
                        if success:
                            buffer.tofile(save_path)
                        else:
                            print(f"Error saving eye to {save_path}")
                        
                        eyes_extracted += 1

                processed_count += 1
        print(f"Finished {dataset}. Images with eyes detected: {processed_count}. Total eyes extracted: {eyes_extracted}")

if __name__ == '__main__':
    main()
