import cv2
import yaml
import numpy as np
from src.detector.hand_detector import HandDetector
from src.detector.calibration import Calibrator
from src.measurement.dimension_calculator import DimensionCalculator
from src.visualization.drawer import Drawer

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_object_pixels(frame):
    """
    Calculate reference object pixels with improved accuracy for fixed distance
    """
    height, width = frame.shape[:2]
    # Using 17% of frame width as reference for credit card at 50cm distance
    return int(width * 0.17)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    detector = HandDetector(config)
    calibrator = Calibrator()
    calculator = DimensionCalculator(calibrator)
    drawer = Drawer(config, calibrator)  # Fixed: Added calibrator parameter
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
    
    print("\n=== Hand Measurement System ===")
    print("Instructions:")
    print("1. Posisikan kamera tepat 50cm dari objek")
    print("2. Tahan kartu kredit atau ID card secara horizontal")
    print("3. Tekan 'c' untuk kalibrasi menggunakan kartu")
    print("4. Setelah kalibrasi, tunjukkan tangan untuk pengukuran")
    print("5. Tekan 'q' untuk keluar\n")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        results = detector.detect(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate dimensions if calibrated
                dimensions = calculator.get_hand_dimensions(hand_landmarks)
                
                # Draw visualization with measurements if calibrated
                frame = drawer.draw_frame(frame, hand_landmarks, dimensions)
        
        # Show calibration status and distance reminder
        status = calibrator.get_calibration_status()
        if status['is_calibrated']:
            cv2.putText(frame, "Calibrated", (1110, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Add distance reminder
            cv2.putText(frame, "50cm", (1110, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Calibrated", (1050, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Add distance instruction
            cv2.putText(frame, "Set 50cm", (1050, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Hand Measurement System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            reference_pixels = calculate_object_pixels(frame)
            if calibrator.calibrate(reference_pixels):
                print("\nKalibrasi berhasil pada jarak 50cm!")
                print("Anda dapat melanjutkan pengukuran tangan")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()