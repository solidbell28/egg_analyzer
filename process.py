from typing import Dict, Tuple
import numpy as np
from ultralytics import YOLO
import cv2


def process_eggs(
    config: Dict,
    model: YOLO,
    image: np.ndarray
) -> Tuple[np.ndarray, int, int]:

    brightness_threshold = config['brightness_threshold']
    conf_threshold = config['conf_threshold']

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_image.shape[:2]

    results = model.predict(rgb_image, conf=conf_threshold, verbose=False)

    white_count = 0
    dark_count = 0

    annotated_img = rgb_image.copy()

    if results[0].masks is not None:
        for i, mask in enumerate(results[0].masks):
            bin_mask = mask.data[0].cpu().numpy()
            bin_mask = (bin_mask * 255).astype(np.uint8)
            bin_mask = cv2.resize(bin_mask, (orig_w, orig_h))

            masked_region = cv2.bitwise_and(
                rgb_image, rgb_image, mask=bin_mask)
            mean_brightness = np.mean(masked_region[bin_mask == 255])

            if mean_brightness > brightness_threshold:
                white_count += 1
                color = (0, 255, 0)
            else:
                dark_count += 1
                color = (0, 0, 255)

            x1, y1, x2, y2 = map(
                int, results[0].boxes[i].xyxy[0].cpu().numpy())

            points = mask.xy[0].astype(np.int32)
            cv2.polylines(annotated_img, [points], True, color, 2)

            cv2.putText(
                annotated_img,
                f"{'White' if mean_brightness > brightness_threshold else 'Dark'}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    return annotated_img_bgr, white_count, dark_count
