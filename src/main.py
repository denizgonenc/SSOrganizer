from pathlib import Path

import torch
import PIL

data_dir = Path("data")

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Your GPU model

for img_path in data_dir.glob("*.png"):
    print(img_path.name)  # filename only
    print(img_path.parent)  # folder path