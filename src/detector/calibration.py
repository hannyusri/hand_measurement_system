class Calibrator:
    def __init__(self):
        # Menggunakan rasio default yang lebih realistis
        self.pixel_to_cm_ratio = None
        self.is_calibrated = False
        self.reference_object_length_cm = 8.56  # Panjang kartu standar (cm)
        self.last_reference_pixels = None
        
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
            
        # Simpan referensi pixel terakhir
        self.last_reference_pixels = reference_pixels
            
        # Hitung rasio pixel ke cm
        self.pixel_to_cm_ratio = reference_cm / reference_pixels
        self.is_calibrated = True
        
        print(f"Kalibrasi selesai:")
        print(f"- Reference pixels: {reference_pixels}")
        print(f"- Reference cm: {reference_cm}")
        print(f"- Ratio: 1 pixel = {self.pixel_to_cm_ratio:.6f} cm")
        return True
        
    def pixels_to_cm(self, pixels):
        """
        Konversi pixels ke centimeter dengan validasi
        """
        if pixels is None or pixels == 0:
            return 0.0
            
        if not self.is_calibrated:
            print("Warning: System not calibrated, using default ratio")
            return 0.0
            
        # Validasi input
        if isinstance(pixels, (int, float)):
            return float(pixels) * self.pixel_to_cm_ratio
        return 0.0
        
    def get_calibration_status(self):
        """
        Mendapatkan status kalibrasi saat ini
        """
        return {
            'is_calibrated': self.is_calibrated,
            'ratio': self.pixel_to_cm_ratio,
            'last_reference_pixels': self.last_reference_pixels,
            'reference_cm': self.reference_object_length_cm
        }