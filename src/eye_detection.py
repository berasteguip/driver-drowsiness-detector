import cv2

eye_cascade = cv2.CascadeClassifier('../haarcascade_eye.xml')
webcam = cv2.VideoCapture(0)

while True:
    _, img = webcam.read()

    img = cv2.flip(img, 1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 2, 4)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)

    cv2.imshow('Eye detection', img)
    key = cv2.waitKey(10)

    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()