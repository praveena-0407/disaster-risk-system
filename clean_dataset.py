import os
from PIL import Image

dataset_path = "dataset/Comprehensive Disaster Dataset(CDD)"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

removed = 0

for category in os.listdir(dataset_path):
    category_folder = os.path.join(dataset_path, category)
    if not os.path.isdir(category_folder):
        continue

    for root, dirs, files in os.walk(category_folder):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            try:
                img = Image.open(filepath)
                img.verify()
            except Exception:
                print(f"❌ Removing bad image: {filepath}")
                os.remove(filepath)
                removed += 1

print(f"\n✅ Done! Removed {removed} corrupted images.")