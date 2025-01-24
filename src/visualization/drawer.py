import cv2
import mediapipe as mp
import numpy as np
import math

class Drawer:
    def __init__(self, config):
        self.config = config
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.colors = {
            'thumb': (0, 120, 255),     # Biru
            'index': (0, 255, 0),       # Hijau
            'middle': (0, 0, 255),      # Merah
            'ring': (255, 0, 255),      # Magenta
            'pinky': (255, 255, 0),     # Cyan
            'forearm': (255, 165, 0)    # Oranye
        }
        self.finger_names = {
            'thumb': 'Ibu Jari',
            'index': 'Telunjuk',
            'middle': 'Jari Tengah',
            'ring': 'Jari Manis',
            'pinky': 'Kelingking',
            'forearm': 'Lengan Bawah'
        }

    def create_info_panel(self, frame, dimensions):
        """Membuat panel informasi dengan gaya Windows"""
        if dimensions is None:
            return frame
            
        height, width, _ = frame.shape
        panel_width = 300
        
        # Buat panel semi-transparan dengan warna abu-abu gelap
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), 
                     (panel_width, 330), (40, 40, 40), -1)
        
        # Tambahkan transparansi
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Header panel
        cv2.rectangle(frame, (0, 0), (panel_width, 35), (60, 60, 60), -1)
        cv2.putText(frame, "PENGUKURAN TANGAN & LENGAN", 
                   (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Tampilkan hasil pengukuran
        y_position = 50
        
        # Tampilkan pengukuran lengan bawah terlebih dahulu
        if 'forearm_length_cm' in dimensions:
            color = self.colors['forearm']
            length_cm = dimensions['forearm_length_cm']
            
            cv2.rectangle(frame, 
                        (10, y_position - 2),
                        (20, y_position + 8),
                        color, -1)
            
            text1 = "Lengan Bawah"
            text2 = f"Panjang: {length_cm:.1f} cm"
            
            cv2.putText(frame, text1,
                       (30, y_position + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (200, 200, 200), 1)
            cv2.putText(frame, text2,
                       (30, y_position + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (180, 180, 180), 1)
                       
            y_position += 55

        # Tampilkan pengukuran jari-jari
        for finger_name, indo_name in self.finger_names.items():
            if finger_name != 'forearm':
                # Cek ketersediaan data pengukuran
                length_key = f'{finger_name}_length_cm'
                tip_dip_key = f'{finger_name}_tip_to_dip_cm'
                
                if length_key in dimensions and tip_dip_key in dimensions:
                    length_cm = dimensions[length_key]
                    tip_to_dip_cm = dimensions[tip_dip_key]
                    color = self.colors[finger_name]
                    
                    cv2.rectangle(frame, 
                                (10, y_position - 2),
                                (20, y_position + 8),
                                color, -1)
                    
                    text1 = f"{indo_name}"
                    text2 = f"Ke pergelangan: {length_cm:.1f} cm"
                    text3 = f"Ke ruas pertama: {tip_to_dip_cm:.1f} cm"
                    
                    cv2.putText(frame, text1,
                               (30, y_position + 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (200, 200, 200), 1)
                    cv2.putText(frame, text2,
                               (30, y_position + 28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (180, 180, 180), 1)
                    cv2.putText(frame, text3,
                               (30, y_position + 48),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (180, 180, 180), 1)
                               
                    y_position += 55

        return frame

    def draw_measurement_lines(self, frame, hand_landmarks, dimensions):
        """Menggambar garis pengukuran"""
        if hand_landmarks and dimensions:
            height, width, _ = frame.shape
            
            # Gambar garis lengan bawah
            if 'forearm_points' in dimensions:
                wrist = dimensions['forearm_points']['wrist']
                forearm_end = dimensions['forearm_points']['end']
                
                wrist_px = (int(wrist.x * width), int(wrist.y * height))
                forearm_end_px = (int(forearm_end.x * width), int(forearm_end.y * height))
                
                self.draw_gradient_line(frame, wrist_px, forearm_end_px, self.colors['forearm'], 2)
                cv2.circle(frame, wrist_px, 4, self.colors['forearm'], -1)
                cv2.circle(frame, forearm_end_px, 4, self.colors['forearm'], -1)
            
            wrist = hand_landmarks.landmark[0]
            wrist_px = (int(wrist.x * width), int(wrist.y * height))
            
            finger_tips = {
                'thumb': 4,
                'index': 8,
                'middle': 12,
                'ring': 16,
                'pinky': 20
            }
            
            finger_dips = {
                'thumb': 3,
                'index': 7,
                'middle': 11,
                'ring': 15,
                'pinky': 19
            }
            
            for finger_name in finger_tips.keys():
                tip = hand_landmarks.landmark[finger_tips[finger_name]]
                dip = hand_landmarks.landmark[finger_dips[finger_name]]
                
                tip_px = (int(tip.x * width), int(tip.y * height))
                dip_px = (int(dip.x * width), int(dip.y * height))
                color = self.colors[finger_name]
                
                cv2.line(frame, wrist_px, tip_px, color, 1)
                self.draw_dashed_line(frame, tip_px, dip_px, color)
                
                cv2.circle(frame, tip_px, 3, color, -1)
                cv2.circle(frame, dip_px, 3, color, -1)

    def draw_gradient_line(self, frame, start_point, end_point, color, thickness):
        """Menggambar garis dengan efek gradien"""
        num_steps = 50
        for i in range(num_steps):
            alpha = i / float(num_steps)
            x = int(start_point[0] * (1 - alpha) + end_point[0] * alpha)
            y = int(start_point[1] * (1 - alpha) + end_point[1] * alpha)
            
            intensity = int(255 * (1 - alpha * 0.3))
            current_color = (
                min(color[0] + 30, intensity),
                min(color[1] + 30, intensity),
                min(color[2] + 30, intensity)
            )
            
            cv2.circle(frame, (x, y), thickness, current_color, -1)

    def draw_dashed_line(self, frame, p1, p2, color, dash_length=5):
        """Menggambar garis putus-putus"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx * dx + dy * dy)
        dx = dx / dist
        dy = dy / dist
        
        curr_x = p1[0]
        curr_y = p1[1]
        step = dash_length * 2
        
        for i in range(0, int(dist), step):
            x1 = int(curr_x)
            y1 = int(curr_y)
            x2 = int(curr_x + dx * dash_length)
            y2 = int(curr_y + dy * dash_length)
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)
            curr_x += dx * step
            curr_y += dy * step

    def draw_landmarks(self, frame, hand_landmarks):
        """Gambar landmark dengan tampilan minimal"""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(
                color=(180, 180, 180),
                thickness=1,
                circle_radius=1
            ),
            self.mp_draw.DrawingSpec(
                color=(140, 140, 140),
                thickness=1
            )
        )

    def draw_frame(self, frame, hand_landmarks, dimensions):
        """Fungsi utama untuk menggambar semua elemen"""
        if hand_landmarks:
            # Gambar landmark
            self.draw_landmarks(frame, hand_landmarks)
            
            # Gambar garis pengukuran
            self.draw_measurement_lines(frame, hand_landmarks, dimensions)
            
            # Tambahkan panel informasi
            frame = self.create_info_panel(frame, dimensions)
            
        return frame