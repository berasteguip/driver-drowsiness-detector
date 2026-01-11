import cv2
import numpy as np

class BasePreprocessor:
    """
    Clase base con la lógica de recorte cuadrado, padding y normalización.
    """
    def __init__(
        self,
        output_size,
        margin,
        use_clahe,
        pad_mode
    ):
        self.output_size = output_size
        self.margin = margin
        self.use_clahe = use_clahe
        self.pad_mode = pad_mode

        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

    def __call__(self, img, box):
        if box is None:
            return None

        # Si img ya es gris (2D), obtenemos H, W directamente. Si es BGR (3D), img.shape[:2]
        H, W = img.shape[:2]
        x, y, w, h = box

        # 1) Centro de la caja original
        cx = x + w // 2
        cy = y + h // 2

        # 2) Lado del cuadrado (con margen)
        side = int(max(w, h) * (1.0 + self.margin))

        # 3) Coordenadas del cuadrado ideal
        x0 = cx - side // 2
        y0 = cy - side // 2
        x1 = cx + side // 2
        y1 = cy + side // 2

        # 4) Padding necesario (cuánto nos salimos de la imagen)
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        # 5) Clamp al frame (coordenadas válidas dentro de la imagen)
        x0_valid = max(0, x0)
        y0_valid = max(0, y0)
        x1_valid = min(W, x1)
        y1_valid = min(H, y1)

        # Recorte de la parte válida
        crop = img[y0_valid:y1_valid, x0_valid:x1_valid]

        # 6) Aplicar padding si hace falta para recuperar el tamaño "side"
        if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
            if self.pad_mode == "replicate":
                crop = cv2.copyMakeBorder(
                    crop,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    borderType=cv2.BORDER_REPLICATE
                )
            else:  # constant
                crop = cv2.copyMakeBorder(
                    crop,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=128
                )

        # 7) Asegurar Escala de grises
        # Si la imagen tenía 3 canales, convertimos. Si ya era gris, la dejamos.
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # 8) Normalización de iluminación (CLAHE)
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 9) Resize final (sin distorsión, porque ya hicimos el recorte cuadrado)
        output = cv2.resize(
            gray,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR
        )

        return output
    

class FacePreprocessor(BasePreprocessor):
    """
    Preprocesador específico para detectar y guardar Caras (128x128 por defecto).
    """
    def __init__(self, output_size=128, margin=0.15, use_clahe=True, pad_mode="replicate"):
        super().__init__(output_size, margin, use_clahe, pad_mode)


class EyePreprocessor(BasePreprocessor):
    """
    Preprocesador específico para Ojos.
    Tamaño por defecto reducido (32x32) ideal para HOG.
    use_clahe=False por defecto si asumimos que venimos de una cara ya procesada (para no contrastar dos veces),
    pero se puede activar si venimos de RAW.
    """
    def __init__(self, output_size=32, margin=0.15, use_clahe=False, pad_mode="replicate"):
        super().__init__(output_size, margin, use_clahe, pad_mode)
