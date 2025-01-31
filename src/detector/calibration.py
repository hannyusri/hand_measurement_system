class Calibrator:
    def __init__(self):
        self.pixel_to_cm_ratio = None
        self.is_calibrated = False
        self.reference_object_length_cm = 8.56  # Panjang kartu standar (cm)
        self.last_reference_pixels = None
        self.camera_distance_cm = 50  # Jarak tetap kamera ke objek
        self.focal_length = None      # Focal length kamera (akan dihitung saat kalibrasi)
        
    def calibrate(self, reference_pixels, reference_cm=None):
        """
        Kalibrasi menggunakan objek referensi dengan jarak tetap
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
        
        # Hitung focal length menggunakan rumus: F = (P x D) / W
        # Dimana: F = focal length, P = ukuran dalam pixel, D = jarak ke objek, W = ukuran sebenarnya
        self.focal_length = (reference_pixels * self.camera_distance_cm) / reference_cm
        
        # Hitung rasio pixel ke cm berdasarkan focal length
        self.pixel_to_cm_ratio = reference_cm / reference_pixels
        
        # Sesuaikan rasio berdasarkan jarak tetap
        self.pixel_to_cm_ratio *= (self.camera_distance_cm / 50)  # Normalisasi ke jarak 50cm
        
        self.is_calibrated = True
        
        print(f"Kalibrasi selesai:")
        print(f"- Reference pixels: {reference_pixels}")
        print(f"- Reference cm: {reference_cm}")
        print(f"- Camera distance: {self.camera_distance_cm} cm")
        print(f"- Focal length: {self.focal_length:.2f}")
        print(f"- Ratio: 1 pixel = {self.pixel_to_cm_ratio:.6f} cm at {self.camera_distance_cm}cm distance")
        return True
        
    def pixels_to_cm(self, pixels, distance_cm=None):
        """
        Konversi pixels ke centimeter dengan kompensasi jarak
        Args:
            pixels: Ukuran dalam pixel
            distance_cm: Jarak aktual ke objek (opsional, default menggunakan jarak tetap)
        """
        if pixels is None or pixels == 0:
            return 0.0
            
        if not self.is_calibrated:
            print("Warning: System not calibrated, using default ratio")
            return 0.0
            
        if not isinstance(pixels, (int, float)):
            return 0.0
            
        if distance_cm is None:
            distance_cm = self.camera_distance_cm
            
        # Gunakan focal length untuk menghitung ukuran sebenarnya
        # Rumus: W = (P x D) / F
        # Dimana: W = ukuran sebenarnya, P = ukuran dalam pixel, D = jarak ke objek, F = focal length
        actual_cm = (pixels * distance_cm) / self.focal_length
        return float(actual_cm)
        
    def get_calibration_status(self):
        """
        Mendapatkan status kalibrasi saat ini
        """
        return {
            'is_calibrated': self.is_calibrated,
            'ratio': self.pixel_to_cm_ratio,
            'last_reference_pixels': self.last_reference_pixels,
            'reference_cm': self.reference_object_length_cm,
            'camera_distance': self.camera_distance_cm,
            'focal_length': self.focal_length
        }