import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class ShapeName(Enum):
    TRIANGULO = "TRIANGULO"
    CUADRADO = "CUADRADO"
    RECTANGULO = "RECTANGULO"
    PENTAGONO = "PENTAGONO"
    CIRCULO = "CIRCULO" 

@dataclass
class DetectedShape:
    """Clase de datos para transferir la información de una figura detectada."""
    label: ShapeName
    box: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[int, int]
    contour: np.ndarray

class ShapeDetector:
    """
    Detector de formas geométricas robustas para Visión por Ordenador.
    Figuras soportadas: Triángulo, Cuadrado, Rectángulo, Pentágono, Círculo.
    """

    def __init__(self, min_area: int = 3000, epsilon_factor: float = 0.05):
        """
        :param min_area: Área mínima en píxeles para considerar una figura (filtra ruido).
        :param epsilon_factor: Precisión de la aproximación poligonal (0.03 - 0.05 es estándar).
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor

    def detect_all(self, image: np.ndarray) -> List[DetectedShape]:
        """
        Procesa la imagen y detecta todas las figuras válidas presentes.
        """
        if image is None:
            return []

        # 1. Preprocesamiento (Gris -> Blur -> Umbral)
        processed = self._preprocess(image)

        # 2. Extracción de contornos
        # Usamos RETR_EXTERNAL para ignorar agujeros dentro de las figuras
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            # Filtrar ruido por tamaño
            if cv2.contourArea(c) < self.min_area:
                continue

            # Identificar la forma
            shape_label = self._identify_shape(c)
            
            if shape_label:
                # Calcular metadatos para visualización (Bounding Box y Centroide)
                x, y, w, h = cv2.boundingRect(c)
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, shape_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w // 2, y + h // 2

                detection = DetectedShape(
                    label=shape_label,
                    box=(x, y, w, h),
                    centroid=(cX, cY),
                    contour=c
                )
                results.append(detection)

        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convierte a escala de grises, suaviza y binariza la imagen."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # IMPORTANTE: 
        # Usa THRESH_BINARY_INV si tus figuras son NEGRAS sobre fondo BLANCO (papel).
        # Usa THRESH_BINARY si tus figuras son BLANCAS sobre fondo OSCURO.
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def _identify_shape(self, contour: np.ndarray) -> Optional[str]:
        peri = cv2.arcLength(contour, True)
        # Bajamos un poco el epsilon para que el círculo no se "aplaste" a cuadrado tan fácil
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        num_vertices = len(approx)
    
        # 1. CÁLCULO DE CIRCULARIDAD (Independiente de los vértices)
        area = cv2.contourArea(contour)
        # Fórmula: 4 * pi * Area / Perímetro^2
        if peri > 0:
            circularity = (4 * np.pi * area) / (peri ** 2)
        else:
            circularity = 0

        # 2. PRIORIDAD: Si es muy redondo, ES UN CÍRCULO (aunque tenga 4 vértices)
        if circularity > 0.82: # Un círculo perfecto es 1.0, un cuadrado es ~0.78
            return "CIRCULO"

        # 3. Si no es redondo, clasificamos por vértices
        if num_vertices == 3:
            return "TRIANGULO"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            return "CUADRADO" if 0.90 <= ar <= 1.10 else "RECTANGULO"
        elif num_vertices == 5:
            if cv2.isContourConvex(approx):
                return "PENTAGONO"
            
        return None