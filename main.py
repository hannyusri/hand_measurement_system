import cv2
import numpy as np
from detector import HandArmDetector
from measurement import Measurement
from storage import HandMeasurementStorage

def main():
    url = "https://192.168.1.63:8080/video" # url dari kamera device, disini menggunakan hp andoid dengan aplikasi IP Webcam (playstore)
    cap = cv2.VideoCapture(url)
    
    detector = HandArmDetector()
    measurement = Measurement()
    storage = HandMeasurementStorage()
    storage.load_from_file()
    
    calibrated = False
    current_hand = 1
    measuring = False
    measurement_complete = False
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Tidak dapat menerima frame. Mencoba kembali...")
            cap.release()
            cap = cv2.VideoCapture(url)
            continue

        height, width = frame.shape[:2]
        crop_percent = 20
        crop_height = int(height * crop_percent / 100)
        crop_width = int(width * crop_percent / 100)
        frame = frame[crop_height:height-crop_height, crop_width:width-crop_width]

        scale_percent = 70
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Referensi kotak kalibrasi ( bisa menggunakan ktp atau yang sejenis)
        ref_rect_width = 150
        ref_rect_height = 100
        ref_rect_x = int((frame.shape[1] - ref_rect_width) / 2)
        ref_rect_y = int((frame.shape[0] - ref_rect_height) / 2)
        ref_rect = (ref_rect_x, ref_rect_y, ref_rect_width, ref_rect_height)
            
        frame, hand_landmarks = detector.find_landmarks(frame)
        
        # pengkalibrasian dan pengukuran tangan
        if not calibrated:
            cv2.rectangle(frame, (ref_rect[0], ref_rect[1]), 
                         (ref_rect[0] + ref_rect[2], ref_rect[1] + ref_rect[3]),  
                         (0, 255, 0), 2)
            cv2.putText(frame, "Letakkan kartu di kotak hijau", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Tekan 'c' untuk kalibrasi", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Mengukur Tangan #{current_hand}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not measuring:
                cv2.putText(frame, "Tekan 'm' untuk mulai mengukur", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tekan 's' untuk simpan pengukuran", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if hand_landmarks and calibrated and measuring:
            measurements = {}
            
            if len(hand_landmarks) >= 21:
                # Pengukuran lengan bawah menggunakan rasio antropometri
                wrist_point = hand_landmarks[0]  
                palm_base = hand_landmarks[9]    
                pinky_base = hand_landmarks[17]  
                
                # Hitung lebar telapak tangan (dari pangkal kelingking ke pangkal jempol)
                palm_width = measurement.calculate_distance(hand_landmarks[1], hand_landmarks[17])
                
                # ukur panjang lengan bawah menggunakan lebar telapak tangan yaitu 3.5x
                forearm_ratio = 3.5
                
                if palm_width:
                    estimated_forearm_length = palm_width * forearm_ratio
                    
                    # Hitung vektor arah dari tengah telapak ke pergelangan
                    direction = np.array([wrist_point[0] - palm_base[0], 
                                        wrist_point[1] - palm_base[1]])
                    direction = direction / np.linalg.norm(direction)
                    
                    # Hitung titik akhir lengan bawah berdasarkan estimasi panjang
                    forearm_end = np.array(wrist_point) + direction * estimated_forearm_length
                    
                    # Gambar garis lengan bawah
                    cv2.line(frame, 
                            (int(wrist_point[0]), int(wrist_point[1])),
                            (int(forearm_end[0]), int(forearm_end[1])),
                            (0, 255, 0), 2)
                    
                    # Perhitungan akhir lengan bawah
                    forearm_length = measurement.calculate_distance(
                        wrist_point, [int(forearm_end[0]), int(forearm_end[1])])
                    if forearm_length:
                        measurements["Panjang Lengan Bawah"] = f"{forearm_length / 10:.1f}"

                # Pengukuran lebar telapak (dari ibu jari ke kelingking)
                palm_width = measurement.calculate_distance(hand_landmarks[4], hand_landmarks[20])
                if palm_width:
                    measurements["Lebar Telapak (Jempol-JariKelingking)"] = f"{palm_width / 10:.1f}"
            
                # Pengukuran tiap ruas berdasarkan landmark jari
                finger_data = {
                    'Ibu Jari': [(1,2), (2,3), (3,4)], 
                    'Telunjuk': [(5,6), (6,7), (7,8)],
                    'Tengah': [(9,10), (10,11), (11,12)],
                    'Manis': [(13,14), (14,15), (15,16)],
                    'Kelingking': [(17,18), (18,19), (19,20)]
                }

                for finger_name, segments in finger_data.items():
                    finger_measurements = {}
                    
                    # Mengukur panjang total jari
                    if finger_name == 'Ibu Jari':
                        total_length = measurement.calculate_distance(hand_landmarks[1], hand_landmarks[4])
                    else:
                        base_idx = segments[0][0]
                        tip_idx = segments[-1][1]
                        total_length = measurement.calculate_distance(hand_landmarks[base_idx], hand_landmarks[tip_idx])
                    
                    if total_length:
                        finger_measurements["Total"] = f"{total_length / 10:.1f}"
                    
                    # Mengukur setiap ruas jari
                    for idx, (start, end) in enumerate(segments, 1):
                        segment_length = measurement.calculate_distance(
                            hand_landmarks[start], hand_landmarks[end])
                        if segment_length:
                            finger_measurements[f"Ruas {idx}"] = f"{segment_length / 10:.1f}"
                    
                    # Menyimpan pengukuran jari
                    measurements[finger_name] = finger_measurements
            
                # Tampilkan pengukuran
                y_pos = 30  #jarak vertikal teks ke frame
                for label, value in measurements.items():
                    if isinstance(value, dict):
                        # Untuk pengukuran jari yang memiliki multiple values
                        text = f"{label}:"
                        cv2.putText(frame, text, (10, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_pos += 25
                        
                        for segment_label, segment_value in value.items():
                            text = f"  {segment_label}: {segment_value} cm"
                            (text_width, text_height), _ = cv2.getTextSize(text, 
                                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (10, y_pos - text_height - 5), 
                                        (10 + text_width, y_pos + 5), (0, 0, 0), -1)
                            cv2.putText(frame, text, (10, y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y_pos += 25
                    else:
                        # Untuk pengukuran tunggal
                        text = f"{label}: {value} cm"
                        (text_width, text_height), _ = cv2.getTextSize(text, 
                                                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (10, y_pos - text_height - 5), 
                                    (10 + text_width, y_pos + 5), (0, 0, 0), -1)
                        cv2.putText(frame, text, (10, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_pos += 25
        
        cv2.imshow("Pengukuran Panjang Tangan", frame)
        key = cv2.waitKey(1)
        
        if key == ord('c'):
            measurement.calibrate(frame, ref_rect)
            calibrated = True
            print("Sistem Terkalibrasi!")
            
        elif key == ord('m') and calibrated and not measuring:
            measuring = True
            print(f"Mulai mengukur tangan #{current_hand}")
            
        elif key == ord('s') and measuring:
            if hand_landmarks and len(measurements) > 0:
                storage.add_measurement(measurements, current_hand)
                storage.save_to_file()
                print(f"Pengukuran tangan #{current_hand} disimpan!")
                current_hand += 1
                measuring = False
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
