"""Competition metric helpers and score wrappers."""

from __future__ import annotations

import io
from typing import Optional

import cairosvg
import cv2
import kagglehub
import numpy as np
from PIL import Image, ImageFilter

from .evaluators import initialize_evaluators

try:
    svg_constraints = kagglehub.package_import("metric/svg-constraints")
except Exception:
    svg_constraints = None


class ParticipantVisibleError(Exception):
    pass


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple[int, int] = (384, 384)) -> Image.Image:
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace("<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)


class ImageProcessor:
    """Image-processing chain preserved from notebook metric experiments."""

    def __init__(self, image: Image.Image, seed: Optional[int] = None):
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self

    def apply_median_filter(self, size: int = 3):
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(
        self,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ):
        img_array = np.asarray(self.image)
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency: float = 0.5):
        img_array = np.array(self.image, dtype=np.float32)
        result = np.zeros_like(img_array)
        for channel_idx in range(3):
            f_transform = np.fft.fft2(img_array[:, :, channel_idx])
            shifted = np.fft.fftshift(f_transform)
            rows, cols = img_array[:, :, channel_idx].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            radius = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
            mask[mask_area] = 1

            filtered_shift = shifted * mask
            inverse_shift = np.fft.ifftshift(filtered_shift)
            img_back = np.fft.ifft2(inverse_shift)
            img_back = np.real(img_back)
            result[:, :, channel_idx] = img_back

        result = np.clip(result, 0, 255).astype(np.uint8)
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality: int = 85):
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent: float = 0.05):
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


def evaluate_with_competition_metric(
    svg: str,
    prompt: str,
    device: Optional[str] = None,
) -> dict:
    vqa_evaluator, aesthetic_evaluator = initialize_evaluators(device=device)
    image = svg_to_png(svg)
    vqa_score = 0.0
    aesthetic_score = aesthetic_evaluator.score(image)
    combined_score = aesthetic_score
    return {
        "vqa_score": vqa_score,
        "aesthetic_score": aesthetic_score,
        "combined_score": combined_score,
    }
