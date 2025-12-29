import cv2
import numpy as np

class FacePreprocessor:
    def __init__(
        self,
        output_size=128,
        margin=0.15,
        use_clahe=True,
        pad_mode="replicate"  # "replicate" | "constant"
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

    def __call__(self, img, face_box):
        if face_box is None:
            return None

        H, W = img.shape[:2]
        x, y, w, h = face_box

        # 1) Centro de la cara
        cx = x + w // 2
        cy = y + h // 2

        # 2) Lado del cuadrado (con margen)
        side = int(max(w, h) * (1.0 + self.margin))

        # 3) Coordenadas del cuadrado ideal
        x0 = cx - side // 2
        y0 = cy - side // 2
        x1 = cx + side // 2
        y1 = cy + side // 2

        # 4) Padding necesario
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        # 5) Clamp al frame
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)

        face = img[y0:y1, x0:x1]

        # 6) Aplicar padding si hace falta
        if any(p > 0 for p in [pad_left, pad_top, pad_right, pad_bottom]):
            if self.pad_mode == "replicate":
                face = cv2.copyMakeBorder(
                    face,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    borderType=cv2.BORDER_REPLICATE
                )
            else:  # constant
                face = cv2.copyMakeBorder(
                    face,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=128
                )

        # 7) Escala de grises
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 8) Normalización de iluminación
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 9) Resize final (sin distorsión)
        face_norm = cv2.resize(
            gray,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR
        )

        return face_norm
