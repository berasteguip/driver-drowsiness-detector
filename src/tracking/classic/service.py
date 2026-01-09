from __future__ import annotations

import cv2
import joblib
import time  # <-- Añadido
from pathlib import Path
from detection.detector import Detector 
from processing.eyes_features import HOGFeatureExtractor, HOGConfig
from processing.preprocess import EyePreprocessor

from config import MODELS_DIR

def load_models(models_dir: Path):
    left_model_path = models_dir / "xgb_eye_left.pkl"
    right_model_path = models_dir / "xgb_eye_right.pkl"

    if not left_model_path.exists() or not right_model_path.exists():
        raise FileNotFoundError(f"Modelos no encontrados en {models_dir}.")

    return joblib.load(left_model_path), joblib.load(right_model_path)

def run_classic_tracker():
    models_dir = MODELS_DIR
    
    try:
        model_left, model_right = load_models(models_dir)
        print("Modelos cargados exitosamente.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    detector_facial = Detector()
    eye_preprocessor = EyePreprocessor(output_size=32, margin=0.15, use_clahe=True)
    
    hog_cfg = HOGConfig(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    feature_extractor = HOGFeatureExtractor(hog_cfg)

    cap = cv2.VideoCapture(0)
    
    # --- Variables de estado para el refresco ---
    frame_count = 0
    update_interval = 30  # Cada 30 frames se actualiza la inferencia
    last_status = "INICIALIZANDO..."
    last_prob = 0.0
    text_color = (0, 255, 0) # Verde por defecto

    # Variables para FPS
    prev_time = 0.0
    fps = 0.0

    print(f"Sistema iniciado. La predicción se actualizará cada {update_interval} frames.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # --- FPS ---
        cur_t = time.time()
        if prev_time > 0:
            dt = cur_t - prev_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = (0.9 * fps + 0.1 * inst_fps) if fps > 0 else inst_fps
        prev_time = cur_t

        face_rect, eyes_rects, mouth_rect = detector_facial.detect(frame)
        
        # --- Lógica de Inferencia Automática ---
        if frame_count % update_interval == 0:
            if face_rect is not None and eyes_rects is not None and len(eyes_rects) >= 2:
                # Preprocesar y Predecir
                left_img = eye_preprocessor(frame, eyes_rects[0])
                right_img = eye_preprocessor(frame, eyes_rects[1])

                if left_img is not None and right_img is not None:
                    feat_left = feature_extractor.extract(left_img)
                    feat_right = feature_extractor.extract(right_img)

                    p_left = model_left.predict_proba(feat_left.reshape(1, -1))[0][1]
                    p_right = model_right.predict_proba(feat_right.reshape(1, -1))[0][1]

                    last_prob = (p_left + p_right) / 2.0
                    
                    if last_prob > 0.5:
                        last_status = "CANSADO (DROWSY)"
                        text_color = (0, 0, 255) # Rojo
                    else:
                        last_status = "DESPIERTO (ACTIVE)"
                        text_color = (0, 255, 0) # Verde
            else:
                last_status = "ROSTRO NO DETECTADO"
                text_color = (0, 165, 255) # Naranja

        # --- Dibujo en Pantalla ---
        # 1. Dibujar rectángulos de detección
        if face_rect is not None:
            xf, yf, wf, hf = face_rect
            cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
            if eyes_rects is not None:
                for (ex, ey, ew, eh) in eyes_rects:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        # 2. UI: Rectángulo de fondo para el texto (mejora legibilidad)
        cv2.rectangle(frame, (10, 10), (400, 85), (0, 0, 0), -1)
        
        # 3. UI: Mostrar Estado y Probabilidad
        cv2.putText(frame, f"ESTADO: {last_status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, f"PROB. CANSANCIO: {last_prob:.2%}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 4. UI: Barra de progreso para el siguiente escaneo
        progress = (frame_count % update_interval) / update_interval
        cv2.rectangle(frame, (10, 90), (10 + int(390 * progress), 95), (255, 255, 0), -1)

        # --- Dibujar FPS en esquina superior derecha ---
        fps_text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x = frame.shape[1] - tw - 10
        y = 10 + th
        cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Monitor de Somnolencia", frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_classic_tracker()