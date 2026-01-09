import cv2
from shape_detector import ShapeDetector
import numpy as np

def test(img: np.ndarray):
    
    # Instanciamos el detector una sola vez
    detector = ShapeDetector(min_area=1000)

    print("Presiona 'q' para salir.")

    # 1. DETECCIÓN
    detected_objects = detector.detect_all(img)

    # 2. VISUALIZACIÓN (Separada de la lógica)
    for obj in detected_objects:
        # Dibujar contorno exacto
        cv2.drawContours(img, [obj.contour], -1, (0, 255, 0), 2)
            
        # Dibujar nombre en el centroide
        cv2.putText(img, obj.label, (obj.centroid[0] - 50, obj.centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        # Opcional: Dibujar Bounding Box
        x, y, w, h = obj.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Mostrar conteo total
        cv2.putText(img, f"Detectados: {len(detected_objects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Multi-Shape Detector", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    img = cv2.imread("test_shapes.png")
    test(img)