import numpy as np

class Measurement:
    def __init__(self, reference_width_mm=91):
        self.reference_width = reference_width_mm
        self.pixels_per_metric = None
        
    def calibrate(self, frame, reference_rect):
        self.pixels_per_metric = reference_rect[2] / self.reference_width
        
    def calculate_distance(self, point1, point2):
        if self.pixels_per_metric is None or point1 is None or point2 is None:
            return None
        return np.sqrt(((point1[0] - point2[0]) ** 2) + 
                      ((point1[1] - point2[1]) ** 2)) / self.pixels_per_metric
