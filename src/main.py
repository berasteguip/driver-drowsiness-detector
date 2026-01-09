import sys
import time

# --- IMPORTS DE BLOQUES ---
# Bloque A: Seguridad
from config import validate_paths
from security.shape_auth.service import ShapePassword
from security.hand_auth.service import HandPassword

# Bloque B: Tracking
from tracking.classic.service import run_classic_tracker
from tracking.modern.service import run_modern_tracker

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_user_choice(prompt: str, valid_options: list) -> str:
    while True:
        choice = input(f"{Colors.BLUE}{prompt} {valid_options}: {Colors.ENDC}").strip()
        if choice in valid_options:
            return choice
        print(f"{Colors.WARNING}Opción no válida.{Colors.ENDC}")

def run_security_phase():
    print(f"\n{Colors.BOLD}--- BLOQUE A: SEGURIDAD ---{Colors.ENDC}")
    print("Seleccione método de desbloqueo:")
    print("1. Patrones Geométricos (Formas) [Opción A1]")
    print("2. Gestos Manuales (Manos)       [Opción A2]")
    
    choice = get_user_choice("Opción:", ['1', '2'])
    
    print(f"\n{Colors.GREEN}Iniciando autenticación...{Colors.ENDC}")
    time.sleep(0.5)

    try:
        if choice == '1':
            auth = ShapePassword(["CUADRADO", "CIRCULO", "TRIANGULO", "PENTAGONO"])
            auth.start()
        else:
            auth = HandPassword(["ROCK", "PEACE", "SURF", "VULCAN"])
            auth.start()
            
            
        print(f"\n{Colors.BOLD}{Colors.GREEN}>> ACCESO CONCEDIDO <<{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"\n{Colors.FAIL}Fallo de seguridad: {e}{Colors.ENDC}")
        return False

def run_tracking_phase():
    print(f"\n{Colors.BOLD}--- BLOQUE B: MONITOREO ---{Colors.ENDC}")
    print("Seleccione motor de tracking:")
    print("1. Tracker Clásico (HOG + XGBoost) [Opción B1]")
    print("2. Tracker Moderno (MediaPipe)     [Opción B2]")
    
    choice = get_user_choice("Opción:", ['1', '2'])
    
    print(f"\n{Colors.GREEN}Arrancando motor de visión... (ESC para salir){Colors.ENDC}")
    time.sleep(0.5)
    
    if choice == '1':
        run_classic_tracker()
    else:
        run_modern_tracker()

def main():
    print("Validando archivos del sistema...")
    validate_paths()
    
    print(f"{Colors.HEADER}=== DRIVER DROWSINESS DETECTOR ==={Colors.ENDC}")
    
    try:
        #if not run_security_phase():
        #    sys.exit(1)
            
        run_tracking_phase()
        
    except KeyboardInterrupt:
        print("\nCierre forzado por usuario.")
        sys.exit(0)

if __name__ == '__main__':
    print('0')
    main()