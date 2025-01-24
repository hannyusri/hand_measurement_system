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
    """Menghitung panjang objek referensi dalam pixel"""
    height, width = frame.shape[:2]
    # Menggunakan 15% dari lebar frame sebagai referensi
    return int(width * 0.15)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    detector = HandDetector(config)
    calibrator = Calibrator()
    calculator = DimensionCalculator(calibrator)
    drawer = Drawer(config)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    
    # Baca frame pertama untuk kalibrasi awal
    success, frame = cap.read()
    if success:
        # Gunakan 15% dari lebar frame sebagai referensi
        reference_pixels = calculate_object_pixels(frame)
        # Kalibrasi dengan kartu standar (8.56 cm)
        calibrator.calibrate(reference_pixels)
    
    print("=== Sistem Pengukuran Tangan ===")
    print("Tekan 'c' untuk kalibrasi ulang")
    print("Tekan 'q' untuk keluar")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Detect hands
        results = detector.detect(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate dimensions
                dimensions = calculator.get_hand_dimensions(hand_landmarks)
                
                # Draw everything
                frame = drawer.draw_frame(frame, hand_landmarks, dimensions)
        
        # Show frame
        cv2.imshow('Hand Measurement System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Kalibrasi ulang
            reference_pixels = calculate_object_pixels(frame)
            if calibrator.calibrate(reference_pixels):
                print("Kalibrasi ulang berhasil")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()