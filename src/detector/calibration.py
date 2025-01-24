import numpy as np

class Calibrator:
    def __init__(self):
        self.pixel_to_cm_ratio = 0.026458333  # Default ratio (1 pixel ≈ 0.026458333 cm pada 96 DPI)
        self.is_calibrated = False
        self.reference_object_length_cm = 8.56  # Panjang kartu standar
        
    def calibrate(self, reference_pixels, reference_cm=None):
        """
        Kalibrasi menggunakan objek referensi
        Args:
            reference_pixels: Panjang dalam pixel
            reference_cm: Panjang dalam cm (opsional, default menggunakan kartu standar)
        """
        if reference_cm is None:
            reference_cm = self.reference_object_length_cm
            
        if reference_pixels <= 0:
            print("Error: Nilai pixel harus lebih besar dari 0")
            return False
            
        self.pixel_to_cm_ratio = reference_cm / reference_pixels
        self.is_calibrated = True
        print(f"Kalibrasi selesai: 1 pixel = {self.pixel_to_cm_ratio:.4f} cm")
        return True
        
    def pixels_to_cm(self, pixels):
        """
        Konversi pixels ke centimeter
        Menggunakan ratio default jika belum dikalibrasi
        """
        if pixels is None or pixels == 0:
            return 0.0
            
        return float(pixels) * self.pixel_to_cm_ratio