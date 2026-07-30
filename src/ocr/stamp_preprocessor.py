import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Переводит изображение в grayscale."""
    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Гарантирует трёхканальное BGR-изображение,
    которое ожидает Paddle recognition.
    """
    if image.ndim == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return image


def original(image: np.ndarray) -> np.ndarray:
    """Исходное изображение — baseline."""
    return _to_bgr(image.copy())


def upscale(
    image: np.ndarray,
    scale: int = 3
) -> np.ndarray:
    """Увеличение изображения."""

    result = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    return _to_bgr(result)


def upscale_grayscale(
    image: np.ndarray,
    scale: int = 3
) -> np.ndarray:
    """Увеличение + grayscale."""

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = _to_gray(resized)

    return _to_bgr(gray)


def upscale_otsu(
    image: np.ndarray,
    scale: int = 3
) -> np.ndarray:
    """Увеличение + grayscale + Otsu."""

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = _to_gray(resized)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return _to_bgr(binary)


def upscale_adaptive(
    image: np.ndarray,
    scale: int = 3
) -> np.ndarray:
    """Увеличение + grayscale + adaptive threshold."""

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = _to_gray(resized)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return _to_bgr(binary)


def get_preprocessing_methods():
    return {
        "original": original,
        "upscale": upscale,
        "grayscale": upscale_grayscale,
        "otsu": upscale_otsu,
        "adaptive": upscale_adaptive,
    }