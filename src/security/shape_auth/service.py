import cv2
import numpy as np
import time  # <-- Añadido
from .shape_detector import ShapeDetector
from typing import List

class ShapePassword:
    def __init__(self, password: List[str]):
        self.password = password
        self.current_step = 0  # Índice de la figura que estamos buscando
        self.stable_counter = 0 # Contador para confirmar que la detección no es ruido
        self.REQUIRED_STABILITY = 15 # Frames consecutivos que debe aparecer la figura

    def start(self):
        detector = ShapeDetector(min_area=3000) # Aumentamos área para ignorar fondo
        cap = cv2.VideoCapture(0)
        
        print(f"Sistema de Seguridad Iniciado. Buscando: {self.password[0]}")

        # Variables para FPS
        prev_time = 0.0
        fps = 0.0

        while self.current_step < len(self.password):
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

            target_shape = self.password[self.current_step]
            detected_shapes = detector.detect_all(frame)
            
            # Buscamos si la figura que toca está en el frame
            found_target = any(s.label == target_shape for s in detected_shapes)

            if found_target:
                self.stable_counter += 1
            else:
                self.stable_counter = 0 # Si desaparece un frame, reseteamos (anti-ruido)

            # Si la figura ha sido estable, avanzamos al siguiente patrón
            if self.stable_counter >= self.REQUIRED_STABILITY:
                print(f"¡{target_shape} detectado con éxito!")
                self.current_step += 1
                self.stable_counter = 0
                if self.current_step < len(self.password):
                    print(f"Siguiente figura: {self.password[self.current_step]}")

            # --- Feedback Visual ---
            self._draw_ui(frame, detected_shapes, target_shape)
            
            # Dibujar FPS en esquina superior derecha
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x = frame.shape[1] - tw - 10
            y = 10 + th
            cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Sistema de Seguridad", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

        if self.current_step == len(self.password):
            print("ACCESO GARANTIZADO")
            # Aquí llamarías al siguiente bloque: tracker() 
        
        cap.release()
        cv2.destroyAllWindows()

    def _draw_ui(self, frame, shapes, target):
        # Dibujar todas las figuras detectadas
        for s in shapes:
            color = (0, 255, 0) if s.label == target else (0, 0, 255)
            cv2.drawContours(frame, [s.contour], -1, color, 2)
            cv2.putText(frame, s.label, (s.centroid[0], s.centroid[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Barra de progreso para el patrón actual
        progress = int((self.stable_counter / self.REQUIRED_STABILITY) * 200)
        cv2.rectangle(frame, (50, 400), (250, 420), (50, 50, 50), -1)
        cv2.rectangle(frame, (50, 400), (50 + progress, 420), (0, 255, 0), -1)
        cv2.putText(frame, f"Validando {target}...", (50, 390), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


if __name__ == "__main__":
    
    password = ShapePassword()
    password.start()