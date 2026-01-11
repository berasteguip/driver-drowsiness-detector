# Driver Drowsiness Detector

Este proyecto es un sistema de visión por computadora diseñado para detectar la somnolencia del conductor en tiempo real. Integra módulos de **autenticación biométrica** y dos motores de **monitoreo de fatiga** configurables.

## 📋 Tabla de Contenidos
- [Instalación](#-instalación)
- [Ejecución](#-ejecución)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)

---

## ⚙️ Instalación

El sistema es compatible con **Python 3.11.10**.

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd driver-drowsiness-detector
Configurar el entorno virtual:

Bash

python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
Instalar dependencias:

Bash

pip install -r requirements.txt
Las librerías principales incluyen OpenCV, MediaPipe, XGBoost y Scikit-learn.

🚀 Ejecución
El sistema se gestiona desde el script principal, que coordina la transición entre la fase de seguridad y la de monitoreo:

Bash

python src/main.py
Al iniciar, el programa validará que los archivos críticos (como los modelos .pkl) existan en las rutas configuradas antes de mostrar el menú de opciones.

🧠 Arquitectura del Sistema
El flujo de trabajo se divide en dos bloques principales:

1. Bloque A: Seguridad (Autenticación)
Antes de activar el tracker, el usuario debe superar un desafío de seguridad:

A1 - Patrones Geométricos (Shape Auth): Utiliza OpenCV para detectar contornos y clasificar formas geométricas (Triángulo, Cuadrado, Círculo, etc.). El usuario debe presentar una secuencia específica frente a la cámara que sea estable por al menos 15 frames.

A2 - Gestos Manuales (Hand Auth): Utiliza MediaPipe Hands para identificar signos manuales como ROCK, PEACE o VULCAN. La entrada se valida contra una lista predefinida para conceder el acceso.

2. Bloque B: Monitoreo (Tracking)
Tras la autenticación, se selecciona el motor de detección de somnolencia:

B1 - Tracker Clásico (XGBoost): Emplea Haar Cascades para detectar el rostro y los ojos. Extrae características mediante HOG (Histogram of Oriented Gradients) y utiliza un modelo XGBoost para predecir la probabilidad de cansancio basándose en el estado del ojo. La inferencia se refresca cada 30 frames para optimizar el rendimiento.

B2 - Tracker Moderno (MediaPipe): Utiliza la malla facial de MediaPipe para obtener 468 puntos clave. Calcula métricas geométricas precisas como:

EAR (Eye Aspect Ratio): Para detectar el parpadeo y ojos cerrados.

MAR (Mouth Aspect Ratio): Para identificar bostezos.

PERCLOS: Calcula el porcentaje de tiempo que los ojos permanecen cerrados en una ventana de 60 segundos para determinar fatiga acumulada.

driver-drowsiness-detector/
├── models/                  # Modelos XGBoost (.pkl) y Haar Cascades (.xml)
├── src/
│   ├── main.py              # Orquestador del sistema
│   ├── config.py            # Gestión de rutas absolutas y validación
│   ├── calibration.py       # Calibración de cámara mediante tablero de ajedrez
│   ├── security/            # Módulos de autenticación por gestos y formas
│   ├── tracking/            # Implementaciones de trackers (Classic vs Modern)
│   ├── detection/           # Detectores faciales y de componentes
│   └── processing/          # Preprocesamiento de imágenes y extracción de features
└── requirements.txt         # Lista de dependencias y versiones
Notas Técnicas
Calibración: src/calibration.py utiliza funciones de OpenCV para obtener la matriz intrínseca y coeficientes de distorsión de la cámara.

Procesamiento: El sistema incluye herramientas en src/processing/data_processing.py para normalizar imágenes de rostros y ojos antes del entrenamiento o inferencia.