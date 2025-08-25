from pathlib import Path
from typing import List

from PIL import Image

import numpy as np

BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"

loaded_paths = []

def load_batch(data_dir: Path, batch_size: int):
    """
        Generator that yields:
            A list of loaded Pillow images
            A list of paths of said images
    """

    batch = []
    paths = []

    for img_path in data_dir.rglob("*.png"):
        img = Image.open(img_path).convert("RGB")
        batch.append(img)
        paths.append(img_path)

        if len(batch) == batch_size:
            yield batch, paths
            batch = []
            paths = []

    # Yield any remaining images in the last batch
    if batch:
        yield batch, paths


def preprocess_batch(batch: List[Image], img_size: tuple = (224, 224), mean=None, std=None) -> np.ndarray:
    """
        Resize and normalize a list of PIL Images into a NumPy array (B, H, W, 3).
    """

    processed = []
    for img in batch:
        img = img.resize(img_size)
        arr = np.array(img) / 255.0
        # Specific normalization if preprocessed model demands it
        if mean is not None and std is not None:
            arr = (arr - mean) / std
        processed.append(arr)
    return np.stack(processed)


for batch, paths in load_batch(data_dir, batch_size=32):
    loaded_paths.extend(paths)
    batch_np = preprocess_batch(batch)
    print(batch_np.shape)