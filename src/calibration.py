import cv2
from typing import List
import numpy as np
import imageio
import copy
import glob
import matplotlib.pyplot as plt

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    
    arr = np.zeros((chessboard_shape[0]*chessboard_shape[1], 3), np.float32)
    for i in range(chessboard_shape[0]):
        for j in range(chessboard_shape[1]):
            arr[j*chessboard_shape[0]+i] = [np.float32(j*dx), np.float32(i*dy), np.float32(0)]
    return arr

# todos los archivos .png en la carpeta data/left
imgs_path = [img_path for img_path in glob.glob("../imgs/calibration/*.jpg")]
imgs = load_images(imgs_path)

corners = [cv2.findChessboardCorners(img, (4, 6)) for img in imgs]

corners_copy = copy.deepcopy(corners)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(i, cor[1], (4,6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]


# --------------------------
# DIBUJAR ESQUINAS
imgs_copy = copy.deepcopy(imgs)

for i in range(len(imgs_copy)):
    if corners[i][0]:
        cv2.drawChessboardCorners(imgs_copy[i], (4, 6), corners_refined[i], True)
        cv2.imshow("img", imgs_copy[i])
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        print(key)
    if key == ord('q'):
        break        



chessboard_points = np.array([get_chessboard_points((4, 6), 0.0315, 0.0315) for _ in range(len(corners))])

valid_corners = [corners_refined[i] for i in range(len(imgs_copy)) if corners[i][0]]
valid_corners = np.asarray(valid_corners, dtype=np.float32)

print("ImagePoints shape  -> ", valid_corners.shape)
print("ObjectPoints shape -> ", chessboard_points.shape)


rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, imgs_gray[1].shape, np.zeros((3,3)), None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

print('rms: ', rms)
print('extrinsics: ', extrinsics)

