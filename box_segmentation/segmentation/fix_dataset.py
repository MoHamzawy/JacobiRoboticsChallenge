import os
import json
import cv2
import yaml

def fix_json_image_sizes(json_path: str, image_root: str, output_json_path: str = None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors_fixed = []

    for image_info in data["images"]:
        file_name = image_info["file_name"]
        json_width = image_info["width"]
        json_height = image_info["height"]

        image_path = os.path.join(image_root, file_name)

        if not os.path.exists(image_path):
            print(f"❌ File not found: {file_name}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not open image: {file_name}")
            continue

        actual_height, actual_width = img.shape[:2]

        if (json_width, json_height) != (actual_width, actual_height):
            errors_fixed.append((file_name, (json_width, json_height), (actual_width, actual_height)))
            image_info["width"] = actual_width
            image_info["height"] = actual_height

    if errors_fixed:
        print(f"🔧 Fixed {len(errors_fixed)} mismatched entries:")
        for file_name, old, new in errors_fixed:
            print(f" - {file_name}: {old} -> {new}")
    else:
        print("✅ All entries already match actual image sizes.")

    output_path = output_json_path or json_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved fixed annotations to: {output_path}")

def main():
    with open("segmentation/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["root_path"]

    splits = [
        ("train", cfg["train_json"], cfg["train_images"]),
        ("val", cfg["val_json"], cfg["val_images"]),
    ]

    for name, json_rel, images_rel in splits:
        print(f"\n🔍 Processing {name} set...")
        json_path = os.path.join(root, json_rel)
        image_dir = os.path.join(root, images_rel)
        fix_json_image_sizes(json_path, image_dir)

if __name__ == "__main__":
    main()
