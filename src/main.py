import sys
import time

# --- BLOCK IMPORTS ---
# Block A: Security
from config import validate_paths
from security.shape_auth.service import ShapePassword
from security.hand_auth.service import HandPassword

# Block B: Tracking
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
        print(f"{Colors.WARNING}Invalid option.{Colors.ENDC}")

def run_security_phase():
    print(f"\n{Colors.BOLD}--- BLOCK A: SECURITY ---{Colors.ENDC}")
    print("Select unlock method:")
    print("1. Geometric Patterns (Shapes) [Option A1]")
    print("2. Hand Gestures (Hands)       [Option A2]")
    
    choice = get_user_choice("Option:", ['1', '2'])
    
    print(f"\n{Colors.GREEN}Starting authentication...{Colors.ENDC}")
    time.sleep(0.5)

    try:
        if choice == '1':
            auth = ShapePassword(["SQUARE", "CIRCLE", "TRIANGLE", "PENTAGON"])
            auth.start()
        else:
            auth = HandPassword(["ROCK", "PEACE", "SURF", "VULCAN"])
            auth.start()
            
            
        print(f"\n{Colors.BOLD}{Colors.GREEN}>> ACCESS GRANTED <<{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"\n{Colors.FAIL}Security failure: {e}{Colors.ENDC}")
        return False

def run_tracking_phase():
    print(f"\n{Colors.BOLD}--- BLOCK B: MONITORING ---{Colors.ENDC}")
    print("Select tracking engine:")
    print("1. Classic Tracker (HOG + XGBoost) [Option B1]")
    print("2. Modern Tracker (MediaPipe)      [Option B2]")
    
    choice = get_user_choice("Option:", ['1', '2'])
    
    print(f"\n{Colors.GREEN}Starting vision engine... (ESC to exit){Colors.ENDC}")
    time.sleep(0.5)
    
    if choice == '1':
        run_classic_tracker()
    else:
        run_modern_tracker()

def is_complete_mode(args: list) -> bool:
    return any(arg.lower() == "complete" for arg in args)

def main():
    print("Validating system files...")
    validate_paths()
    
    print(f"{Colors.HEADER}=== DRIVER DROWSINESS DETECTOR ==={Colors.ENDC}")
    
    try:
        complete_mode = is_complete_mode(sys.argv[1:])
        if complete_mode:
            if not run_security_phase():
                sys.exit(1)
        else:
            print(f"{Colors.WARNING}Fast mode: skipping the security block.{Colors.ENDC}")
            
        run_tracking_phase()
        
    except KeyboardInterrupt:
        print("\nForced shutdown by user.")
        sys.exit(0)

if __name__ == '__main__':
    main()
