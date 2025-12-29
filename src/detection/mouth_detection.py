import os
import cv2

class MouthDetector:
    '''
    Clase que detecta la boca en una imagen y devuelve coordenadas globales
    '''
    def __init__(self):
        # Cargamos el clasificador de boca
        xml_path = os.path.join(os.path.dirname(__file__), '../../haarcascade_mcs_mouth.xml')
        self.mouth_cascade = cv2.CascadeClassifier(xml_path)

    def detect(self, img, face_frame):
        # 1. Convertimos a gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Extraemos coordenadas de la cara: face_frame = (x_cara, y_cara, w_cara, h_cara)
        xf, yf, wf, hf = face_frame
        gray_face = gray[yf :yf+hf, xf:xf+wf]
        
        # 3. Detectamos bocas en la región de la cara
        mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.1, 5)
        
        # Si no detecta ninguna boca, devolvemos None
        if len(mouths) == 0:
            return None
            
        # 4. Ordenamos por área y tomamos la más grande
        sorted_mouths = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)
        mx, my, mw, mh = sorted_mouths[0] # Coordenadas relativas a la cara
        
        # 5. AJUSTE: Sumamos la posición de la cara para obtener coordenadas globales
        # Esto permite que la boca coincida con la imagen original
        global_mouth = (xf + mx, yf + my, mw, mh)
        
        return global_mouth