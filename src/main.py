from tracker import drowsiness_tracker
from security.password import security_system
from mediapipe_tracker import mediapipe_tracker


def main():

    security_system(["ROCK", "PEACE", "SURF", "VULCAN"])

    print("Arrancando tracker...")
    drowsiness_tracker()

    print("Arrancando tracker mediapipe...")
    mediapipe_tracker() # TODO: detectar cansancio o no en función de parámetros

if __name__ == '__main__':
    main()
