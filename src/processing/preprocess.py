import cv2
import numpy as np

class BasePreprocessor:
    """
    Base class with square crop logic, padding, and normalization.
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

        # If img is already grayscale (2D), use H, W directly. If BGR (3D), use img.shape[:2]
        H, W = img.shape[:2]
        x, y, w, h = box

        # 1) Center of the original box
        cx = x + w // 2
        cy = y + h // 2

        # 2) Square side length (with margin)
        side = int(max(w, h) * (1.0 + self.margin))

        # 3) Ideal square coordinates
        x0 = cx - side // 2
        y0 = cy - side // 2
        x1 = cx + side // 2
        y1 = cy + side // 2

        # 4) Required padding (how far we go outside the image)
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        # 5) Clamp to frame (valid coordinates inside the image)
        x0_valid = max(0, x0)
        y0_valid = max(0, y0)
        x1_valid = min(W, x1)
        y1_valid = min(H, y1)

        # Crop the valid region
        crop = img[y0_valid:y1_valid, x0_valid:x1_valid]

        # 6) Apply padding if needed to recover the target "side" size
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

        # 7) Ensure grayscale
        # If the image had 3 channels, convert it. If it was already gray, keep it.
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # 8) Lighting normalization (CLAHE)
        if self.use_clahe:
            gray = self.clahe.apply(gray)

        # 9) Final resize (no distortion since we already cropped to square)
        output = cv2.resize(
            gray,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR
        )

        return output
    

class FacePreprocessor(BasePreprocessor):
    """
    Specific preprocessor to detect and save Faces (128x128 by default).
    """
    def __init__(self, output_size=128, margin=0.15, use_clahe=True, pad_mode="replicate"):
        super().__init__(output_size, margin, use_clahe, pad_mode)


class EyePreprocessor(BasePreprocessor):
    """
    Specific preprocessor for Eyes.
    Default small size (32x32) ideal for HOG.
    use_clahe=False by default if we assume we come from an already processed face
    (to avoid applying contrast twice), but it can be enabled if we come from RAW.
    """
    def __init__(self, output_size=32, margin=0.15, use_clahe=False, pad_mode="replicate"):
        super().__init__(output_size, margin, use_clahe, pad_mode)
