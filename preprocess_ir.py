import cv2
import numpy as np


def preprocess_ir(frame, use_clahe=True, denoise=False):
    """OV9281 흑백/USB 영상을 YOLO 입력으로 변환"""

    if frame is None:
        raise ValueError("입력 이미지가 없습니다.")

    # 1. 흑백으로 변환
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # 2. 8비트 영상으로 정리
    if gray.dtype != np.uint8:
        max_value = float(np.max(gray))
        if max_value > 0:
            gray = np.clip(
                gray.astype(np.float32) / max_value * 255,
                0, 255
            ).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    # 3. 선택적 노이즈 제거
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. CLAHE 국소 대비 향상
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4)
        )
        gray = clahe.apply(gray)

    # 5. YOLO 입력용 3채널 변환
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
