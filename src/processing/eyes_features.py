# eyes_features.py (adapted for Windows Unicode paths)
# Reads images with np.fromfile + cv2.imdecode to avoid cv2.imread failures
# Saves .npz to: <FINAL_PROJECT>/data/features/{left,right}/{active,drowsy}/features_hog.npz

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog


@dataclass(frozen=True)
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"


class HOGFeatureExtractor:
    def __init__(self, cfg: HOGConfig = HOGConfig()):
        self.cfg = cfg

    def extract(self, img_gray: np.ndarray) -> np.ndarray:
        if img_gray is None:
            raise ValueError("img_gray is None")
        if img_gray.ndim != 2:
            raise ValueError(f"Expected grayscale 2D image, got shape {img_gray.shape}")

        feat = hog(
            img_gray,
            orientations=self.cfg.orientations,
            pixels_per_cell=self.cfg.pixels_per_cell,
            cells_per_block=self.cfg.cells_per_block,
            block_norm=self.cfg.block_norm,
            visualize=False,
            feature_vector=True,
        )
        return feat.astype(np.float32)


def project_root_from_this_file() -> Path:
    # .../Final Project/driver-drowsiness-detector/src/processing/features.py
    return Path(__file__).resolve().parents[3]


def list_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files: List[Path] = []
    if not folder.exists():
        return files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def read_gray_unicode_safe(path: Path) -> np.ndarray | None:
    """
    Unicode-safe image read for Windows paths:
    - Uses np.fromfile + cv2.imdecode
    - Returns grayscale 2D array or None
    """
    try:
        stream = np.fromfile(str(path), dtype=np.uint8)
        if stream.size == 0:
            return None
        img = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def save_npz(out_path: Path, X: np.ndarray, y: np.ndarray, paths: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, paths=np.array(paths, dtype=object))


def build_and_save_eye_features(
    hog_cfg: HOGConfig = HOGConfig(),
    strict_size: Tuple[int, int] | None = None,
) -> None:
    root = project_root_from_this_file()
    data_root = root / "data"

    eyes_root = data_root / "eyes"
    out_root = data_root / "features"

    extractor = HOGFeatureExtractor(hog_cfg)

    class_to_label = {"active": 0, "drowsy": 1}
    sides = ["left", "right"]
    classes = ["active", "drowsy"]

    for side in sides:
        for cls in classes:
            in_dir = eyes_root / cls / side
            files = list_images(in_dir)

            if len(files) == 0:
                print(f"[WARN] No images found: {in_dir}")
                continue

            X_list: List[np.ndarray] = []
            y_list: List[int] = []
            path_list: List[str] = []

            n_failed = 0
            for fp in files:
                img = read_gray_unicode_safe(fp)
                if img is None:
                    n_failed += 1
                    continue

                if strict_size is not None and img.shape[:2] != strict_size:
                    raise ValueError(
                        f"Image {fp} has size {img.shape[:2]}, expected {strict_size}."
                    )

                feat = extractor.extract(img)
                X_list.append(feat)
                y_list.append(class_to_label[cls])
                path_list.append(str(fp))

            if len(X_list) == 0:
                print(f"[WARN] All reads failed: {in_dir} (failed={n_failed})")
                continue

            X = np.vstack(X_list).astype(np.float32)
            y = np.array(y_list, dtype=np.int64)

            out_path = out_root / side / cls / "features_hog.npz"
            save_npz(out_path, X, y, path_list)

            print(
                f"[OK] {cls}/{side} -> {out_path} | X={X.shape} | y={y.shape} | n={len(y)} | failed_reads={n_failed}"
            )


if __name__ == "__main__":
    build_and_save_eye_features(
        hog_cfg=HOGConfig(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        ),
        strict_size=(32, 32),
    )
