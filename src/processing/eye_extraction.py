import cv2
import numpy as np
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
        
        # Crear directorio de salida si no existe
        os.makedirs(output_path, exist_ok=True)

        # Buscar imágenes (png o jpg)
        image_files = glob.glob(os.path.join(input_path, "*.png")) + \
                      glob.glob(os.path.join(input_path, "*.jpg"))
        
        print(f"Procesando {len(image_files)} imágenes de '{dataset}' en '{input_path}'...")
        
        processed_count = 0
        eyes_extracted = 0

        for img_file in tqdm(image_files):
            # Usar numpy + imdecode para soportar rutas con tildes/eñes en Windows
            # cv2.imread falla silenciosamente con caracteres no ASCII en la ruta
            try:
                stream = np.fromfile(img_file, dtype=np.uint8)
                img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            except Exception:
                print(f"Error al cargar la imagen: {img_file}")
                img = None

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
                        # Guardar ojo usando imencode y tofile para soportar rutas con tildes/eñes
                        
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
                            print(f"Error al guardar el ojo en {save_path}")
                        
                        eyes_extracted += 1

                processed_count += 1
        print(f"Finalizado {dataset}. Imágenes con ojos detectados: {processed_count}. Total ojos extraídos: {eyes_extracted}")

if __name__ == '__main__':
    main()