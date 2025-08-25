from pathlib import Path
from PIL import Image

import numpy as np

BASE_DIR = Path(__file__).parent.parent
data_dir = BASE_DIR / "data"

loaded_paths = []

def load_batch(data_dir: Path, batch_size: int):
    """
        Generator that yields batches of images as NumPy arrays.
    """

    batch = []
    paths = []

    for img_path in data_dir.rglob("*.png"):
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)  # shape: (H, W, 3)
        batch.append(arr)
        paths.append(img_path)

        if len(batch) == batch_size:
            yield np.stack(batch), paths  # stack into (B, H, W, 3)
            batch = []
            paths = []

    # Yield any remaining images in the last batch
    if batch:
        yield np.stack(batch), paths


for batch, paths in load_batch(data_dir, batch_size=32):
    loaded_paths.extend(paths)