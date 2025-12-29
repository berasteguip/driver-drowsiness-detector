import cv2
import os
import glob
from tqdm import tqdm
from ..detection.eye_detection import EyeDetector
from .preprocess import EyePreprocessor

def main():
    # Inicializamos el detector de ojos
    eye_detector = EyeDetector()
    
    # Inicializamos el preprocesador de ojos (32x32 para HOG, sin CLAHE extra porque ya viene procesado)
    eye_preprocessor = EyePreprocessor(output_size=32, margin=0.1, use_clahe=False)

    # Definimos las rutas
    DATA_DIR = '..\..\data'
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'eyes')

    datasets = ['active', 'drowsy']

    for dataset in datasets:
        input_path = os.path.join(PROCESSED_DIR, dataset)
        output_path = os.path.join(OUTPUT_DIR, dataset)
        
        # Crear directorio de salida si no existe
        os.makedirs(output_path, exist_ok=True)

        # Buscar imágenes (png o jpg)
        image_files = glob.glob(os.path.join(input_path, "*.png")) + \
                      glob.glob(os.path.join(input_path, "*.jpg"))
        
        print(f"Procesando {len(image_files)} imágenes de '{dataset}' en '{input_path}'...")
        
        processed_count = 0
        eyes_extracted = 0

        for img_file in tqdm(image_files):
            img = cv2.imread(img_file)
            if img is None:
                continue
            
            # Como la imagen YA ES LA CARA (processed), el frame de la cara es toda la imagen
            h, w = img.shape[:2]
            face_frame = (0, 0, w, h)

            # Detectar ojos
            eyes_coords = eye_detector.detect(img, face_frame)

            if eyes_coords is not None:
                # Extraer cada ojo usando el preprocesador
                for i, eye_box in enumerate(eyes_coords):
                    
                    # Usamos el preprocesador para extraer y normalizar (32x32)
                    eye_img = eye_preprocessor(img, eye_box)
                    
                    if eye_img is not None:
                        # Guardar ojo
                        base_name = os.path.basename(img_file)
                        name_parts = os.path.splitext(base_name)
                        save_name = f"{name_parts[0]}_eye{i+1}.png"
                        save_path = os.path.join(output_path, save_name)
                        
                        cv2.imwrite(save_path, eye_img)
                        eyes_extracted += 1
                
                processed_count += 1
            
        print(f"Finalizado {dataset}. Imágenes con ojos detectados: {processed_count}. Total ojos extraídos: {eyes_extracted}")

if __name__ == '__main__':
    main()