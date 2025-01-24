import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, config):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config['detection']['max_num_hands'],
            min_detection_confidence=config['detection']['min_detection_confidence'],
            min_tracking_confidence=config['detection']['min_tracking_confidence']
        )
        
    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results