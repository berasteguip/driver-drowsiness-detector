# features.py
# Extract HOG features from preprocessed face images stored in:
#   data/processed/active/
#   data/processed/drowsy/
#
# Produces:
#   X: (N, D) float32
#   y: (N,) int64   (active=0, drowsy=1)
#
# Optional: run as script to save .npz.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

    def extract(self, face_norm_gray: np.ndarray) -> np.ndarray:
        """
        face_norm_gray: 2D grayscale image (H,W), ideally uint8.
        Returns: 1D float32 feature vector.
        """
        if face_norm_gray is None:
            raise ValueError("face_norm_gray is None")

        if face_norm_gray.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {face_norm_gray.shape}")

        feat = hog(
            face_norm_gray,
            orientations=self.cfg.orientations,
            pixels_per_cell=self.cfg.pixels_per_cell,
            cells_per_block=self.cfg.cells_per_block,
            block_norm=self.cfg.block_norm,
            visualize=False,
            feature_vector=True,
        )
        return feat.astype(np.float32)


def _list_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files: List[Path] = []
    if not folder.exists():
        return files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def _read_grayscale(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def build_feature_dataset(
    processed_root: str | Path = "../data/processed",
    class_to_label: Dict[str, int] | None = None,
    hog_cfg: HOGConfig = HOGConfig(),
    strict_size: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Reads all images from class folders under processed_root, extracts HOG features.

    processed_root structure example:
      ../data/processed/active/*.png
      ../data/processed/drowsy/*.png

    class_to_label default:
      {"active": 0, "drowsy": 1}

    strict_size:
      If provided (H,W), will enforce all images are exactly that size; otherwise raise.

    Returns:
      X, y, paths (paths aligned with rows of X)
    """
    root = Path(processed_root)

    if class_to_label is None:
        class_to_label = {
            "active": 0,
            "drowsy": 1,
        }

    extractor = HOGFeatureExtractor(hog_cfg)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    path_list: List[str] = []

    for class_name, label in class_to_label.items():
        class_dir = root / class_name
        files = _list_images(class_dir)
        if not files:
            continue

        for fp in files:
            img = _read_grayscale(fp)
            if img is None:
                continue

            if strict_size is not None:
                Hs, Ws = strict_size
                if img.shape[:2] != (Hs, Ws):
                    raise ValueError(
                        f"Image {fp} has size {img.shape[:2]}, expected {strict_size}. "
                        "Fix your preprocessing or disable strict_size."
                    )

            feat = extractor.extract(img)
            X_list.append(feat)
            y_list.append(int(label))
            path_list.append(str(fp))

    if not X_list:
        raise RuntimeError(
            f"No images found under {root}. Expected class folders like: "
            f"{root/'active'} and {root/'drowsy'} (or names in class_to_label)."
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, path_list


def save_npz(out_path: str | Path, X: np.ndarray, y: np.ndarray, paths: List[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, paths=np.array(paths, dtype=object))


if __name__ == "__main__":
    # Default: build dataset from data/processed/{active,drowsy} and save to data/features_hog.npz
    X, y, paths = build_feature_dataset(
        processed_root="../data/processed",
        class_to_label={"active": 0, "drowsy": 1},
        hog_cfg=HOGConfig(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),
        strict_size=(128, 128),  # set to None if your processed images are not all same size yet
    )
    print(f"X shape: {X.shape} | y shape: {y.shape} | n_active={(y==0).sum()} | n_drowsy={(y==1).sum()}")
    save_npz("../data/features_hog.npz", X, y, paths)
    print("Saved: ../data/features_hog.npz")
