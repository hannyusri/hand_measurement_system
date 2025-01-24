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
        self.buffer_size = 5
        
        # Definisi indeks landmark
        self.finger_tips = {
            'thumb': 4,      # Ibu jari
            'index': 8,      # Telunjuk
            'middle': 12,    # Jari tengah
            'ring': 16,      # Jari manis
            'pinky': 20      # Kelingking
        }
        self.finger_dips = {
            'thumb': 3,      # Ibu jari
            'index': 7,      # Telunjuk
            'middle': 11,    # Jari tengah
            'ring': 15,      # Jari manis
            'pinky': 19      # Kelingking
        }
        self.forearm_landmarks = {
            'wrist': 0,      # Pergelangan tangan
            'wrist_end': 9   # Titik referensi di pergelangan
        }

    def add_to_buffer(self, measurement_type, finger_name, value):
        """Menambahkan pengukuran ke buffer"""
        if measurement_type == 'finger_tips':
            if finger_name not in self.measurement_buffer['finger_tips']:
                self.measurement_buffer['finger_tips'][finger_name] = []
            buffer = self.measurement_buffer['finger_tips'][finger_name]
        else:
            buffer = self.measurement_buffer[measurement_type]

        if len(buffer) >= self.buffer_size:
            buffer.pop(0)
        buffer.append(value)

    def get_stable_measurement(self, measurements):
        """Mendapatkan pengukuran stabil dengan filter outlier"""
        if len(measurements) < 3:
            return None
            
        # Hitung standard deviation
        std = stdev(measurements)
        avg = mean(measurements)
        
        # Filter measurements dalam 2 standard deviations
        filtered = [m for m in measurements if abs(m - avg) <= 2 * std]
        return mean(filtered) if filtered else None

    def calculate_distance(self, p1, p2):
        """Menghitung jarak antara dua titik"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def estimate_forearm_endpoint(self, wrist_point, wrist_ref_point, extension_factor=2.5):
        """
        Memperkirakan titik akhir lengan bawah berdasarkan orientasi pergelangan tangan
        """
        dx = wrist_point.x - wrist_ref_point.x
        dy = wrist_point.y - wrist_ref_point.y
        
        end_point_x = wrist_point.x + (dx * extension_factor)
        end_point_y = wrist_point.y + (dy * extension_factor)
        
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        return Point(end_point_x, end_point_y)

    def get_hand_dimensions(self, landmarks):
        """Menghitung dimensi tangan dengan stabilisasi pengukuran"""
        if not landmarks:
            return None
            
        dimensions = {}
        wrist = landmarks.landmark[self.forearm_landmarks['wrist']]
        wrist_ref = landmarks.landmark[self.forearm_landmarks['wrist_end']]
        
        # Estimasi titik akhir lengan bawah
        forearm_end = self.estimate_forearm_endpoint(wrist, wrist_ref)
        
        # Hitung panjang lengan bawah
        forearm_length = self.calculate_distance(wrist, forearm_end)
        forearm_length_cm = self.calibrator.pixels_to_cm(forearm_length)
        self.add_to_buffer('forearm', None, forearm_length_cm)
        
        # Ambil pengukuran stabil untuk lengan bawah
        stable_forearm = self.get_stable_measurement(self.measurement_buffer['forearm'])
        if stable_forearm:
            dimensions['forearm_length'] = forearm_length
            dimensions['forearm_length_cm'] = stable_forearm
            dimensions['forearm_points'] = {
                'wrist': wrist,
                'end': forearm_end
            }
        
        # Hitung jarak dari pergelangan ke setiap ujung jari
        for finger_name, tip_idx in self.finger_tips.items():
            # Ujung jari ke pergelangan
            finger_tip = landmarks.landmark[tip_idx]
            wrist_to_tip = self.calculate_distance(wrist, finger_tip)
            wrist_to_tip_cm = self.calibrator.pixels_to_cm(wrist_to_tip)
            
            self.add_to_buffer('wrist_to_middle', None, wrist_to_tip_cm)
            stable_wrist_to_tip = self.get_stable_measurement(
                self.measurement_buffer['wrist_to_middle']
            )
            
            if stable_wrist_to_tip:
                dimensions[f'{finger_name}_length'] = wrist_to_tip
                dimensions[f'{finger_name}_length_cm'] = stable_wrist_to_tip
            
            # Ujung jari ke ruas pertama (DIP)
            dip_idx = self.finger_dips[finger_name]
            dip = landmarks.landmark[dip_idx]
            tip_to_dip = self.calculate_distance(finger_tip, dip)
            tip_to_dip_cm = self.calibrator.pixels_to_cm(tip_to_dip)
            
            self.add_to_buffer('finger_tips', finger_name, tip_to_dip_cm)
            stable_tip_to_dip = self.get_stable_measurement(
                self.measurement_buffer['finger_tips'][finger_name]
            )
            
            if stable_tip_to_dip:
                dimensions[f'{finger_name}_tip_to_dip'] = tip_to_dip
                dimensions[f'{finger_name}_tip_to_dip_cm'] = stable_tip_to_dip
            
        return dimensions