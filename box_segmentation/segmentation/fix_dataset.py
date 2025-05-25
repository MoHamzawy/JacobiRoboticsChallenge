import os
import json
import cv2

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

    # Save to file
    output_path = output_json_path or json_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved fixed annotations to: {output_path}")

# === USAGE ===
json_path = r"C:/Users/ID_Admin/Downloads/OSCD/coco_carton/oneclass_carton/annotations/instances_train2017.json"
image_root = r"C:/Users/ID_Admin/Downloads/OSCD/coco_carton/oneclass_carton/images/train2017"

# Optional: change output_json_path if you want to save as a separate file
fix_json_image_sizes(json_path, image_root)
