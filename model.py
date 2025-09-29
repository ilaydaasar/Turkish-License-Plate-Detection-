import os
from ultralytics import YOLO

def train_model():
    # DLL hatası için (opsiyonel)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Modeli yükle
    model = YOLO("yolo11m.pt")

    # Eğitim parametreleri
    train_args = dict(
        data="plaka_tanima_oto.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,             # GPU:0, yoksa "cpu"
        name="plaka_final",
        single_cls=True,
        resume=False,
        workers=0,            # Windows'ta hata olursa 0 yap
        optimizer="AdamW",
    )

    # Eğitimi başlat
    results = model.train(**train_args)
    print("✅ Eğitim tamamlandı!")

if __name__ == "__main__":
    train_model()
