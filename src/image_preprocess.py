"""Image preprocessing service for food recognition."""

import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing: resize, crop to plate, remove background."""

    def __init__(self, temp_dir='/tmp/preprocess'):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def preprocess_image(self, image_path: str) -> str:
        """
        Full preprocessing pipeline: resize -> crop -> remove bg.
        Returns path to final PNG file.
        """
        logger.info(f"Starting preprocessing for {image_path}")
        start_time = cv2.getTickCount()

        try:
            # Save PIL image to temp
            with Image.open(image_path) as img:
                temp_path = os.path.join(self.temp_dir, 'original.png')
                img.save(temp_path, 'PNG')

            # Resize
            resized_path = self._resize_image(temp_path, max_size=800)
            if not resized_path:
                raise ValueError("Resize failed")

            # Skip crop for now
            cropped_path = resized_path

            # Remove background
            final_path = self._remove_background(cropped_path)
            if not final_path:
                raise ValueError("Background removal failed")

            total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            logger.info(f"Preprocessing completed in {total_time:.2f}s")

            return final_path

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _resize_image(self, input_path: str, max_size: int = 800) -> str:
        """Resize image, keeping aspect ratio, longest side = max_size."""
        output_path = os.path.join(self.temp_dir, 'resized.png')

        try:
            with Image.open(input_path) as img:
                width, height = img.size
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized.save(output_path, 'PNG')

            logger.debug(".2f")
            return output_path

        except Exception as e:
            logger.error(f"Resize failed: {str(e)}")
            return None

    def _crop_to_plate(self, input_path: str) -> str:
        """Crop to circular plate using HoughCircles. Returns cropped PNG path or None if failed."""
        output_path = os.path.join(self.temp_dir, 'cropped.png')

        try:
            img = cv2.imread(input_path)
            if img is None:
                logger.error("Cannot read image for cropping")
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=50,
                param2=30,
                minRadius=50,
                maxRadius=250
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Take the largest circle (usually the plate)
                circle = max(circles[0], key=lambda c: c[2])

                x, y, r = circle
                # Crop square around circle
                crop_size = r * 2
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(img.shape[1], x + r)
                y2 = min(img.shape[0], y + r)

                cropped = img[y1:y2, x1:x2]
                cv2.imwrite(output_path, cropped)

                logger.debug(".1f")
                return output_path

            return None  # No plate detected

        except Exception as e:
            logger.error(f"Crop failed: {str(e)}")
            return None

    def _remove_background(self, input_path: str) -> str:
        """Remove background using rembg, save PNG with alpha."""
        output_path = os.path.join(self.temp_dir, 'no_bg.png')

        try:
            with open(input_path, 'rb') as f:
                input_bytes = f.read()

            output_bytes = remove(input_bytes)
            with open(output_path, 'wb') as f:
                f.write(output_bytes)

            logger.debug("Background removal completed")
            return output_path

        except Exception as e:
            logger.error(f"Background removal failed: {str(e)}")
            return None
