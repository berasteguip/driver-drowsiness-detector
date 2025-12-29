import cv2

def draw(img, face_box):

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
    else:
        cv2.putText(img, "Face not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)