import cv2
import numpy as np
import os

def detect_signature(image_path: str, debug: bool = False) -> bool:
    if not os.path.exists(image_path):
        print(f"[ERROR] Signature check: Gambar tidak ditemukan di {image_path}")
        return False
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None: return False
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_image = img.copy() if debug else None
        signature_found = False

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0 and area > 0:
                complexity = (perimeter ** 2) / area
                
                min_area = 1000
                max_area = 100000
                min_complexity = 20

                if (min_area < area < max_area) and (complexity > min_complexity):
                    signature_found = True
                    if debug:
                        cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 3)
                elif debug:
                    cv2.drawContours(debug_image, [contour], -1, (0, 0, 255), 2)

        if debug:
            debug_path = image_path.replace('.jpg', '_debug.jpg')
            cv2.imwrite(debug_path, debug_image)
            print(f"✅ Gambar debug disimpan di: {debug_path}")

        return signature_found
        
    except Exception as e:
        print(f"[ERROR] Gagal deteksi tanda tangan: {e}")
        return False