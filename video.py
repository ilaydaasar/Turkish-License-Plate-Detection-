import cv2

cap = cv2.VideoCapture("Traffic Control CCTV.mp4.crdownload")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # veya "XVID"
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print("Video yeniden kaydedildi: output.mp4")
