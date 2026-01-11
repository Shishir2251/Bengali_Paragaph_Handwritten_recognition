import os
import json

DATA_ROOT = os.path.join("raw_data", "converted")

def build_annotations(split):
    images_dir = os.path.join(DATA_ROOT, split, "images")
    labels_dir = os.path.join(DATA_ROOT, split, "labels")

    if not os.path.exists(images_dir):
        raise RuntimeError(f"❌ Missing folder: {images_dir}")

    annotations = []

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        label_path = os.path.join(
            labels_dir,
            img_name.replace(".jpg", ".json")
        )

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("label", "").strip()
        if not text:
            continue

        annotations.append({
            "image": img_name,
            "text": text
        })

    out_file = os.path.join(DATA_ROOT, split, "annotations.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"✅ {split}: {len(annotations)} samples")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        build_annotations(split)
