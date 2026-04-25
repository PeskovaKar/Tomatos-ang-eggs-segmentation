import cv2
import numpy as np
from ultralytics import YOLO

from core.models import AnalysisResult, DetectedObject


class ImageAnalyzer:
    def __init__(
        self,
        model_path: str = "core/weights/best.pt",
        conf: float = 0.2,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names
        print("Model class names:", self.names)

    def analyze(self, image_path: str) -> AnalysisResult:
        image = cv2.imread(image_path)

        if image is None:
            return AnalysisResult(
                status_message="Ошибка: не удалось загрузить изображение"
            )

        results = self.model(image, conf=self.conf, verbose=False)
        result = results[0]

        output_image = image.copy()
        overlay = image.copy()

        objects = []
        red_count = 0
        yellow_count = 0
        egg_count = 0

        boxes_xyxy = None
        class_ids = None
        confs = None
        masks = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()

        if boxes_xyxy is not None and class_ids is not None and confs is not None:
            for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, class_ids, confs)):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls_id)
                conf = float(conf)

                label = self.names.get(cls_id, f"class_{cls_id}")

                if cls_id == 0:
                    egg_count += 1
                    color = (255, 255, 0)
                    short_label = "E"
                elif cls_id == 2:
                    yellow_count += 1
                    color = (0, 255, 255)
                    short_label = "Y"
                elif cls_id == 1:
                    red_count += 1
                    color = (0, 0, 255)
                    short_label = "R"
                else:
                    color = (180, 180, 180)
                    short_label = "U"

                obj_mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask = cv2.resize(
                        mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    obj_mask = mask

                    colored = np.zeros_like(image)
                    colored[:, :] = color

                    blended = cv2.addWeighted(overlay, 0.5, colored, 0.5, 0)
                    overlay[obj_mask > 0] = blended[obj_mask > 0]

                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    output_image,
                    f"{short_label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                objects.append(
                    DetectedObject(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        label=label,
                        mask=obj_mask,
                    )
                )

        output_image = cv2.addWeighted(overlay, 0.45, output_image, 0.55, 0)

        total_count = red_count + yellow_count + egg_count

        return AnalysisResult(
            original_image=image,
            processed_image=output_image,
            objects=objects,
            total_count=total_count,
            red_count=red_count,
            yellow_count=yellow_count,
            egg_count=egg_count,
            status_message=f"Найдено объектов: {total_count}",
        )