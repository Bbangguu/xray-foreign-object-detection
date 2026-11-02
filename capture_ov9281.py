import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from preprocess_ir import preprocess_ir

# 1. 저장된 X-ray YOLOv8n 모델
MODEL_PATH = Path("xguard_yolov8n_best.pt")
assert MODEL_PATH.exists(), f"모델 없음: {MODEL_PATH}"

model = YOLO(str(MODEL_PATH))

# 2. OV9281 USB 카메라 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("USB 카메라 연결 실패")

# 카메라 해상도 및 FPS 요청
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 30)

print("실제 해상도:",
      cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("설정 FPS:", cap.get(cv2.CAP_PROP_FPS))

# 3. 실시간 탐지
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 수신 실패")
        break

    start = time.perf_counter()

    # 원본 카메라 영상
    original = frame.copy()

    # CLAHE 전처리
    processed = preprocess_ir(
        frame,
        use_clahe=True,
        denoise=False
    )

    # YOLOv8n 탐지
    result = model.predict(
        source=processed,
        imgsz=256,
        conf=0.25,
        verbose=False
    )[0]

    elapsed = time.perf_counter() - start

    # 검출 결과 시각화
    display_frame = result.plot()

    cv2.putText(
        display_frame,
        f"Process + AI: {elapsed*1000:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # 원본 및 전처리 결과 확인
    cv2.imshow("OV9281 Original", original)
    cv2.imshow("OV9281 Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
