import math
import numpy as np
from statistics import mean, stdev

class DimensionCalculator:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.measurement_buffer = {
            'finger_tips': {},
            'wrist_to_middle': [],
            'forearm': []
        }
        self.buffer_size = 10  # Increased buffer size for better stability
        
        # Landmark definitions
        self.finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        self.finger_mcp = {  # Base of fingers
            'thumb': 2,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        self.finger_dips = {
            'thumb': 3,
            'index': 7,
            'middle': 11,
            'ring': 15,
            'pinky': 19
        }
        self.wrist_landmarks = {
            'wrist': 0,
            'wrist_end': 9
        }

    def calculate_3d_distance(self, p1, p2):
        """Calculate 3D distance between two landmarks"""
        return math.sqrt(
            (p1.x - p2.x)**2 + 
            (p1.y - p2.y)**2 + 
            (p1.z - p2.z)**2
        ) * 1000  # Convert to millimeters for better precision

    def get_stable_measurement(self, measurements, threshold=2.0):
        """Get stable measurement with improved outlier filtering"""
        if len(measurements) < 5:  # Require more measurements for stability
            return None
            
        # Remove extreme outliers
        sorted_measurements = sorted(measurements)
        q1 = np.percentile(sorted_measurements, 25)
        q3 = np.percentile(sorted_measurements, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        filtered = [m for m in measurements if lower_bound <= m <= upper_bound]
        
        if len(filtered) < 3:
            return None
            
        return mean(filtered)

    def estimate_forearm_endpoint(self, wrist_point, wrist_ref_point, extension_factor=2.0):
        """
        Memperkirakan titik akhir lengan bawah berdasarkan orientasi pergelangan tangan
        dengan faktor ekstensi yang diperbarui
        """
        # Hitung vektor arah dari wrist_ref ke wrist
        dx = wrist_point.x - wrist_ref_point.x
        dy = wrist_point.y - wrist_ref_point.y
        dz = wrist_point.z - wrist_ref_point.z
        
        # Perpanjang vektor dengan extension_factor
        end_point_x = wrist_point.x + (dx * extension_factor)
        end_point_y = wrist_point.y + (dy * extension_factor)
        end_point_z = wrist_point.z + (dz * extension_factor)
        
        class Point3D:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        return Point3D(end_point_x, end_point_y, end_point_z)

    def get_hand_dimensions(self, landmarks):
        """Calculate hand dimensions with improved accuracy"""
        if not landmarks or not self.calibrator.is_calibrated:
            return None
            
        dimensions = {}
        
        # Get wrist point and reference points
        wrist = landmarks.landmark[self.wrist_landmarks['wrist']]
        wrist_ref = landmarks.landmark[self.wrist_landmarks['wrist_end']]
        
        # Calculate and add forearm measurement
        forearm_end = self.estimate_forearm_endpoint(wrist, wrist_ref)
        forearm_length = self.calculate_3d_distance(wrist, forearm_end)
        forearm_length_cm = self.calibrator.pixels_to_cm(forearm_length)
        
        # Add to buffer for stability
        if len(self.measurement_buffer['forearm']) >= self.buffer_size:
            self.measurement_buffer['forearm'].pop(0)
        self.measurement_buffer['forearm'].append(forearm_length_cm)
        
        # Get stable forearm measurement
        stable_forearm = self.get_stable_measurement(self.measurement_buffer['forearm'])
        if stable_forearm:
            dimensions['forearm_length'] = forearm_length
            dimensions['forearm_length_cm'] = stable_forearm
            dimensions['forearm_points'] = {
                'wrist': wrist,
                'end': forearm_end
            }
        
        # Calculate palm width (distance between index and pinky MCP)
        index_mcp = landmarks.landmark[self.finger_mcp['index']]
        pinky_mcp = landmarks.landmark[self.finger_mcp['pinky']]
        palm_width = self.calculate_3d_distance(index_mcp, pinky_mcp)
        palm_width_cm = self.calibrator.pixels_to_cm(palm_width)
        dimensions['palm_width_cm'] = palm_width_cm
        
        # Calculate finger lengths
        for finger_name in self.finger_tips.keys():
            # Full finger length (MCP to tip)
            mcp = landmarks.landmark[self.finger_mcp[finger_name]]
            tip = landmarks.landmark[self.finger_tips[finger_name]]
            
            finger_length = self.calculate_3d_distance(mcp, tip)
            finger_length_cm = self.calibrator.pixels_to_cm(finger_length)
            
            # Store measurement in buffer
            if finger_name not in self.measurement_buffer['finger_tips']:
                self.measurement_buffer['finger_tips'][finger_name] = []
            
            buffer = self.measurement_buffer['finger_tips'][finger_name]
            if len(buffer) >= self.buffer_size:
                buffer.pop(0)
            buffer.append(finger_length_cm)
            
            # Get stable measurement
            stable_length = self.get_stable_measurement(buffer)
            if stable_length:
                dimensions[f'{finger_name}_length_cm'] = stable_length
            
            # Tip to DIP (last segment) length
            dip = landmarks.landmark[self.finger_dips[finger_name]]
            tip_to_dip = self.calculate_3d_distance(tip, dip)
            tip_to_dip_cm = self.calibrator.pixels_to_cm(tip_to_dip)
            
            dimensions[f'{finger_name}_tip_to_dip_cm'] = tip_to_dip_cm
        
        # Calculate palm length (wrist to middle finger MCP)
        middle_mcp = landmarks.landmark[self.finger_mcp['middle']]
        palm_length = self.calculate_3d_distance(wrist, middle_mcp)
        palm_length_cm = self.calibrator.pixels_to_cm(palm_length)
        dimensions['palm_length_cm'] = palm_length_cm
        
        return dimensions