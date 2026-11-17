# src/preprocessing/dcp_pipeline.py
import random
from pathlib import Path
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

CATEGORIES = {
    '20251112_A': 'fog',
    '20251112_B': 'fog',
    '20251130_A': 'fog',
    '20251130_B': 'fog',
    '20250915_Cebin_mlha': 'fog',
    '20250916_Cebin_oblacno': 'cloudy',
    '20251106_Cebin_jasno': 'clear',
}


def compute_dark_channel_fast(
    image_bgr: np.ndarray, patch_size: int = 15
) -> np.ndarray:
    """Вычисляет карту Dark Channel Prior (DCP) по статье He et al."""
    # 1. Ресайз к 500x500 (как в оригинальной статье)
    img_500 = cv2.resize(image_bgr, (500, 500))

    # 2. Попиксельный минимум среди B, G, R
    min_channel = np.min(img_500, axis=2)

    # 3. Скользящее окно 15x15 через эрозию OpenCV
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel  # массив 500x500 uint8 [0..255]


def extract_dcp_stats(dark_channel_uint8: np.ndarray) -> np.ndarray:
    """Вычисляет скалярные метрики по статье He et al.:

    1. Средняя яркость темного канала (нормированная [0..1])
    2. Доля пикселей равных 0 (zero_ratio)
    3. Доля пикселей со значением < 25 (low_ratio)
    """
    total_pixels = dark_channel_uint8.size
    mean_val = np.mean(dark_channel_uint8) / 255.0
    zero_ratio = np.count_nonzero(dark_channel_uint8 == 0) / total_pixels
    low_ratio = np.count_nonzero(dark_channel_uint8 < 25) / total_pixels

    return np.array([mean_val, zero_ratio, low_ratio], dtype='float32')


def build_dcp_map_dataset(
    base_dir='../',
    output_path='data/dataset_dcp_map.npz',
    img_size=(32, 32),
    max_images=4000,
):
    """Сборка 4-канального датасета (BGR + Dark Channel)[cite: 5]."""
    base_path = Path(base_dir)
    samples = []
    for folder, label in CATEGORIES.items():
        files = list((base_path / folder).glob('*.png'))[:max_images:10]
        for f in files:
            samples.append((f, label))

    random.seed(42)
    random.shuffle(samples)

    features, targets = [], []
    for path, label in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue

        dcp = compute_dark_channel_fast(img)

        # Сжатие до рабочего разрешения эксперимента (32x32)
        img_res = cv2.resize(img, img_size).astype('float32') / 255.0
        dcp_res = (
            cv2.resize(dcp, img_size).astype('float32') / 255.0
        )  # Нормализация в [0, 1]

        # Объединение в 4 канала [32, 32, 4]
        combined = np.dstack([img_res, dcp_res])
        features.append(combined)
        targets.append(label)

    features = np.array(features, dtype='float32')
    targets = LabelBinarizer().fit_transform(targets)

    np.savez(output_path, features=features, targets=targets)
    print(
        f' Датасет 4-канального DCP сохранен: {output_path}, форма: {features.shape}'
    )


def build_dcp_stats_dataset(
    base_dir='../',
    output_path='data/dataset_dcp_stats.npz',
    img_size=(32, 32),
    max_images=4000,
):
    """Сборка мультимодального датасета (BGR картинка 32x32x3 + вектор из 3 признаков)."""
    base_path = Path(base_dir)
    samples = []
    for folder, label in CATEGORIES.items():
        files = list((base_path / folder).glob('*.png'))[:max_images:10]
        for f in files:
            samples.append((f, label))

    random.seed(42)
    random.shuffle(samples)

    images, stats_list, targets = [], [], []
    for path, label in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue

        dcp = compute_dark_channel_fast(img)
        stats = extract_dcp_stats(dcp)

        img_res = cv2.resize(img, img_size).astype('float32') / 255.0

        images.append(img_res)
        stats_list.append(stats)
        targets.append(label)

    images = np.array(images, dtype='float32')
    stats_list = np.array(stats_list, dtype='float32')
    targets = LabelBinarizer().fit_transform(targets)

    np.savez(output_path, images=images, stats=stats_list, targets=targets)
    print(
        f' Датасет со статистиками DCP сохранен: {output_path} (картинок: {len(images)})'
    )
    
if __name__ == '__main__':
  # Этот блок выполняется ТОЛЬКО если файл запущен напрямую,
  # но игнорируется, если файл куда-то импортируют
    build_dcp_map_dataset()