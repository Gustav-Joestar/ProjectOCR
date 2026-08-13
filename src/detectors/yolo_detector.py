from pathlib import Path

from ultralytics import YOLO


class YOLODetector:
    DRAWING_CLASS = 0
    STAMP_CLASS = 1

    def __init__(
        self,
        model_path: str | Path = "models/final_detect_model.pt",
        confidence: float = 0.85,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO-модель не найдена: {self.model_path}"
            )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(self, image):
        if image is None or image.size == 0:
            raise ValueError("Передано пустое изображение.")

        result = self.model.predict(
            source=image,
            imgsz=1024,
            conf=self.confidence,
            verbose=False,
        )[0]

        drawings = []
        stamps = []

        if result.boxes is None:
            return {
                "drawings": drawings,
                "stamps": stamps,
            }

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detection = {
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
            }

            if class_id == self.DRAWING_CLASS:
                drawings.append(detection)
            elif class_id == self.STAMP_CLASS:
                stamps.append(detection)

        drawings.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        stamps.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        return {
            "drawings": drawings,
            "stamps": stamps,
        }

    @staticmethod
    def crop(
        image,
        bbox: tuple[int, int, int, int],
    ):
        if image is None or image.size == 0:
            return None

        height, width = image.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    @staticmethod
    def intersection_area(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> int:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        if x2 <= x1 or y2 <= y1:
            return 0

        return (x2 - x1) * (y2 - y1)

    @classmethod
    def find_stamp_for_drawing(
        cls,
        drawing_bbox: tuple[int, int, int, int],
        stamps: list[dict],
    ) -> dict | None:
        best_stamp = None
        best_overlap = 0

        for stamp in stamps:
            overlap = cls.intersection_area(
                drawing_bbox,
                stamp["bbox"],
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_stamp = stamp

        return best_stamp