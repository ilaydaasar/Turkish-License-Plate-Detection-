from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# MODELLERİ YÜKLE
detection_model = YOLO('runs/detect/plaka_final/weights/best.pt')  # Plaka tespiti
ocr_model = YOLO('detect/segment2/weights/best.pt')  # Karakter okuma

# Sınıf isimleri
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 
               'V', 'Y', 'Z']

def test_your_image(image_path):
    
    print(f"🎯 TEST: {image_path}")
    
    # Görseli yükle
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Görsel yüklenemedi: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("1. 🔍 Plaka tespit ediliyor...")
    
    # Detection modeli ile plakayı bul
    det_results = detection_model(image_rgb, conf=0.3)
    
    for i, det_result in enumerate(det_results):
        boxes = det_result.boxes
        if boxes is not None and len(boxes) > 0:
            
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                
                print(f"   ✅ Plaka bulundu! Güven: {confidence:.3f}")
                print(f"   📍 Konum: [{x1}, {y1}, {x2}, {y2}]")
                
                # Plakayı kırp
                plate_crop = image[y1:y2, x1:x2]
                
                # Kırpılmış plakayı kaydet (kontrol için)
                cv2.imwrite('kirpilmis_plaka.jpg', plate_crop)
                print("   📸 Kırpılmış plaka kaydedildi: kirpilmis_plaka.jpg")
                
                # OCR ile karakterleri oku
                print("2. 🔤 Karakterler okunuyor...")
                ocr_results = ocr_model(plate_crop, conf=0.3)
                
                characters = []
                for ocr_result in ocr_results:
                    ocr_boxes = ocr_result.boxes
                    if ocr_boxes is not None and len(ocr_boxes) > 0:
                        
                        for k, ocr_box in enumerate(ocr_boxes):
                            ox1, oy1, ox2, oy2 = map(int, ocr_box.xyxy[0])
                            ocr_conf = ocr_box.conf[0].item()
                            cls_id = int(ocr_box.cls[0])
                            char = class_names[cls_id]
                            
                            characters.append({
                                'char': char,
                                'confidence': ocr_conf,
                                'center_x': (ox1 + ox2) / 2
                            })
                            
                            print(f"      {k+1}. '{char}' - Güven: {ocr_conf:.3f}")
                
                # Karakterleri birleştir
                if characters:
                    characters.sort(key=lambda x: x['center_x'])
                    plate_text = ''.join([char['char'] for char in characters])
                    print(f"3. 🚗 OKUNAN PLAKA: {plate_text}")
                else:
                    print("3. ❌ Hiç karakter okunamadı")
                    plate_text = "OKUNAMADI"
                
                # Sonuçları göster
                display_result(image_rgb, x1, y1, x2, y2, plate_text, confidence)
                
        else:
            print("❌ Hiç plaka tespit edilemedi")

def display_result(image, x1, y1, x2, y2, plate_text, confidence):
    """Sonuçları görselle göster"""
    
    result_image = image.copy()
    
    # Plaka kutusu
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Bilgi kutusu
    cv2.rectangle(result_image, (x1, y1-60), (x2, y1), (0, 255, 0), -1)
    cv2.putText(result_image, f'PLAKA: {plate_text}', 
                (x1+5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(result_image, f'Guven: %{confidence*100:.1f}', 
                (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.axis('off')
    plt.title('Plaka Tanıma Sonucu')
    plt.show()

if __name__ == "__main__":
    
    your_image_path = 'carplate\datasets\coco8\images\val\77.jpg' 
    
    if os.path.exists(your_image_path):
        test_your_image(your_image_path)
    else:
        print(f"❌ Görsel bulunamadı: {your_image_path}")
        print("📂 Mevcut dosyalar:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"   - {file}")