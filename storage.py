from datetime import datetime
import json

class HandMeasurementStorage:
    def __init__(self):   
        self.measurements_history = []
        
    def add_measurement(self, measurements, hand_number):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        measurement_data = {
            "timestamp": timestamp,
            "hand_number": hand_number,
            "measurements": measurements 
        }
        self.measurements_history.append(measurement_data)
        
    def save_to_file(self):
        with open('hand_measurements.json', 'w') as f:
            json.dump(self.measurements_history, f, indent=4)
            
    def load_from_file(self):
        try:
            with open('hand_measurements.json', 'r') as f:
                self.measurements_history = json.load(f)
        except FileNotFoundError:
            self.measurements_history = []
