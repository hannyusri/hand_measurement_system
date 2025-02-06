import cv2
import mediapipe as mp

class HandArmDetector:
    def __init__(self):
        # Setup model deteksi tangan dengan konfigurasi dasar
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Mode video (real-time)
            max_num_hands=1,          # Maksimal 1 tangan dideteksi
            min_detection_confidence=0.7  # confidence 70% biar akurat << lebih tinggi confidence lebih akurat
        )
        self.mp_draw = mp.solutions.drawing_utils # Modul untuk menggambar landmark
        
    def find_landmarks(self, frame):
        # Konversi warna frame ke RGB karena MediaPipe pake format RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Proses deteksi tangan dalam frame
        hand_results = self.hands.process(frame_rgb)
        hand_landmarks = []
        
        # Kalau ada tangan terdeteksi
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                # Ambil koordinat tiap landmark dan tambahkan ID-nya
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Konversi koordinat ke pixel
                    hand_landmarks.append([cx, cy])
                    cv2.putText(frame, str(id), (cx-10, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                
                # Gambar landmarks dan garis antar landmark di frame
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  # Warna landmark
                    self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)   # Warna garis antar landmark
                )
        
        # Return frame yang udah diolah + list koordinat landmarks
        return frame, hand_landmarks