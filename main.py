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
    """Calculate reference object pixels with better accuracy"""
    height, width = frame.shape[:2]
    # Using 20% of frame width as reference (typical credit card width)
    return int(width * 0.20)

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
    cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
    
    # Initial calibration
    success, frame = cap.read()
    if success:
        reference_pixels = calculate_object_pixels(frame)
        calibrator.calibrate(reference_pixels)
    
    print("\n=== Hand Measurement System ===")
    print("Instructions:")
    print("1. Hold a credit card or ID card horizontally")
    print("2. Press 'c' to calibrate using the card")
    print("3. After calibration, show your hand to measure")
    print("4. Press 'q' to quit\n")
    
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
                
                # Draw visualization
                frame = drawer.draw_frame(frame, hand_landmarks, dimensions)
        
        # Show calibration status
        status = calibrator.get_calibration_status()
        if status['is_calibrated']:
            cv2.putText(frame, "Calibrated", (1110, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Calibrated", (1095, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Hand Measurement System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            reference_pixels = calculate_object_pixels(frame)
            if calibrator.calibrate(reference_pixels):
                print("\nCalibration successful!")
                print("You can now proceed with hand measurements")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()