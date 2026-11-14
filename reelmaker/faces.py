"""Face finder for smart 9:16 cropping (OpenCV YuNet detector)."""

from __future__ import annotations

import os
import statistics
from pathlib import Path

from .utils import APP_DIR

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

MODEL = APP_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


class FaceFinder:
    def __init__(self, video: Path):
        import cv2  # noqa: WPS433 - optional heavy import
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(video))
        self.det = None
        if MODEL.exists() and self.cap.isOpened():
            self.det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.75, 0.3, 50)

    @property
    def ok(self) -> bool:
        return self.det is not None

    def _face_x(self, t: float) -> float | None:
        cv2 = self.cv2
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = 640.0 / max(w, h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        fh, fw = frame.shape[:2]
        self.det.setInputSize((fw, fh))
        _, faces = self.det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        # biggest face = the speaker (usually); ignore tiny background faces
        best = max(faces, key=lambda f: f[2] * f[3])
        if best[3] < fh * 0.08:
            return None
        return float((best[0] + best[2] / 2.0) / fw)

    def center_x(self, start: float, end: float, samples: int = 3) -> float | None:
        if not self.ok:
            return None
        span = max(0.0, end - start)
        times = [start + span * (k + 1) / (samples + 1) for k in range(samples)]
        xs = [x for x in (self._face_x(t) for t in times) if x is not None]
        return statistics.median(xs) if xs else None

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:  # pragma: no cover
            pass
