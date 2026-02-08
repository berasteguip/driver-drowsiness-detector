# Driver Drowsiness Detector

This project is a computer vision system designed to detect driver drowsiness in real time. It integrates **biometric authentication** modules and two configurable **fatigue monitoring** engines.

## Table of Contents
- [Installation](#installation)
- [Running](#running)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Technical Notes](#technical-notes)

---

## Installation

The system is compatible with **Python 3.11.10**.

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd driver-drowsiness-detector
   ```
2. Configure the virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Main libraries include OpenCV, MediaPipe, XGBoost, and Scikit-learn.

## Running

The system is managed from the main script, which coordinates the transition between the security phase and the monitoring phase:

```bash
python src/main.py
```

At startup, the program validates that critical files (such as .pkl models) exist in the configured paths before showing the options menu.

## System Architecture

The workflow is divided into two main blocks:

1. Block A: Security (Authentication)
Before enabling the tracker, the user must pass a security challenge:

A1 - Geometric Patterns (Shape Auth): Uses OpenCV to detect contours and classify geometric shapes (Triangle, Square, Circle, etc.). The user must present a specific sequence in front of the camera that remains stable for at least 15 frames.

A2 - Hand Gestures (Hand Auth): Uses MediaPipe Hands to identify hand signs such as ROCK, PEACE, or VULCAN. The input is validated against a predefined list to grant access.

2. Block B: Monitoring (Tracking)
After authentication, the drowsiness detection engine is selected:

B1 - Classic Tracker (XGBoost): Uses Haar Cascades to detect the face and eyes. Extracts features using HOG (Histogram of Oriented Gradients) and uses an XGBoost model to predict fatigue probability based on eye state. Inference refreshes every 30 frames to optimize performance.

B2 - Modern Tracker (MediaPipe): Uses the MediaPipe face mesh to obtain 468 key points. Calculates precise geometric metrics such as:

EAR (Eye Aspect Ratio): To detect blinking and closed eyes.

MAR (Mouth Aspect Ratio): To identify yawns.

PERCLOS: Calculates the percentage of time the eyes remain closed in a 60-second window to determine accumulated fatigue.

## Project Structure

```
driver-drowsiness-detector/
|-- models/                  # XGBoost models (.pkl) and Haar Cascades (.xml)
|-- src/
|   |-- main.py              # System orchestrator
|   |-- config.py            # Absolute path management and validation
|   |-- calibration.py       # Camera calibration using a chessboard
|   |-- security/            # Gesture and shape authentication modules
|   |-- tracking/            # Tracker implementations (Classic vs Modern)
|   |-- detection/           # Face and component detectors
|   `-- processing/          # Image preprocessing and feature extraction
`-- requirements.txt         # Dependency list and versions
```

## Technical Notes

Calibration: src/calibration.py uses OpenCV functions to obtain the camera intrinsic matrix and distortion coefficients.

Processing: The system includes tools in src/processing/data_processing.py to normalize face and eye images before training or inference.
