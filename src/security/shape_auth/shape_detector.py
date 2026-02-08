import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class ShapeName(Enum):
    TRIANGLE = "TRIANGLE"
    SQUARE = "SQUARE"
    RECTANGLE = "RECTANGLE"
    PENTAGON = "PENTAGON"
    CIRCLE = "CIRCLE" 

@dataclass
class DetectedShape:
    """Data class to transfer information about a detected shape."""
    label: ShapeName
    box: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[int, int]
    contour: np.ndarray

class ShapeDetector:
    """
    Robust geometric shape detector for Computer Vision.
    Supported shapes: Triangle, Square, Rectangle, Pentagon, Circle.
    """

    def __init__(self, min_area: int = 3000, epsilon_factor: float = 0.05):
        """
        :param min_area: Minimum area in pixels to consider a shape (filters noise).
        :param epsilon_factor: Polygon approximation precision (0.03 - 0.05 is standard).
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor

    def detect_all(self, image: np.ndarray) -> List[DetectedShape]:
        """
        Process the image and detect all valid shapes present.
        """
        if image is None:
            return []

        # 1. Preprocessing (Gray -> Blur -> Threshold)
        processed = self._preprocess(image)

        # 2. Contour extraction
        # Use RETR_EXTERNAL to ignore holes inside shapes
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            # Filter noise by size
            if cv2.contourArea(c) < self.min_area:
                continue

            # Identify the shape
            shape_label = self._identify_shape(c)
            
            if shape_label:
                # Compute metadata for visualization (Bounding Box and Centroid)
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
        """Convert to grayscale, smooth, and binarize the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # IMPORTANT:
        # Use THRESH_BINARY_INV if your shapes are BLACK on a WHITE background (paper).
        # Use THRESH_BINARY if your shapes are WHITE on a DARK background.
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def _identify_shape(self, contour: np.ndarray) -> Optional[str]:
        peri = cv2.arcLength(contour, True)
        # Lower epsilon slightly so the circle doesn't collapse into a square too easily
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        num_vertices = len(approx)
    
        # 1. CIRCULARITY CALCULATION (independent of vertices)
        area = cv2.contourArea(contour)
        # Formula: 4 * pi * Area / Perimeter^2
        if peri > 0:
            circularity = (4 * np.pi * area) / (peri ** 2)
        else:
            circularity = 0

        # 2. PRIORITY: If it's very round, it's a CIRCLE (even if it has 4 vertices)
        if circularity > 0.82: # A perfect circle is 1.0, a square is ~0.78
            return "CIRCLE"

        # 3. If not round, classify by vertices
        if num_vertices == 3:
            return "TRIANGLE"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            return "SQUARE" if 0.90 <= ar <= 1.10 else "RECTANGLE"
        elif num_vertices == 5:
            if cv2.isContourConvex(approx):
                return "PENTAGON"
            
        return None
