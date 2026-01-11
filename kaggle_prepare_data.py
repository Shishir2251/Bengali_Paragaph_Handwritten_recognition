import os
import json
import random
import shutil
import subprocess

DATASET = "reasat/banglawriting"
RAW_DIR = "raw_data"
OUTPUT_DIR = "data"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

random.seed(42)


def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR, "--unzip"],
        check=True
    )


def extract_text_from_json(meta: dict) -> str:
    """
    BanglaWriting JSON format:
    {
      "shapes": [
        { "label": "বাংলা লেখা ..." }
      ]
    }
    """
    if "shapes" in meta and isinstance(meta["shapes"], list):
        texts = []
        for shape in meta["shapes"]:
            if isinstance(shape, dict):
                label = shape.get("label", "")
                if isinstance(label, str) and label.strip():
                    texts.append(label.strip())
        return " ".join(texts)

    return ""


def collect_samples():
    samples = []
    scanned = 0

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTS):
                continue

            scanned += 1
            img_path = os.path.join(root, file)
            json_path = os.path.join(root, os.path.splitext(file)[0] + ".json")

            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            text = extract_text_from_json(meta)
            if not text:
                continue

            samples.append({
                "image_path": img_path,
                "image_name": file,
                "text": text
            })

    print(f"🔍 Images scanned: {scanned}")
    print(f"✅ Valid samples found: {len(samples)}")

    if not samples:
        raise RuntimeError(
            "❌ No valid image-json pairs found.\n"
            "👉 JSON uses shapes[].label (check dataset integrity)."
        )

    return samples


def prepare_folders():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)


def main():
    download_dataset()
    prepare_folders()

    samples = collect_samples()
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_data = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    for split, items in split_data.items():
        split_dir = os.path.join(OUTPUT_DIR, split)
        annotations = []

        for item in items:
            shutil.copy(item["image_path"], os.path.join(split_dir, item["image_name"]))
            annotations.append({
                "image": item["image_name"],
                "text": item["text"]
            })

        with open(
            os.path.join(split_dir, "annotations.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        print(f"✅ {split}: {len(items)} samples")

    print("\n🎉 Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
