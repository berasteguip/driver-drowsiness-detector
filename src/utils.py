import cv2
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def draw(img, face_box):

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
    else:
        cv2.putText(img, "Face not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    auc_val = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
