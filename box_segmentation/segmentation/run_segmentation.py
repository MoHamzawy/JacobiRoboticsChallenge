import os
import cv2
import yaml
import random
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


def load_config(path="segmentation/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_predictor(cfg_data):
    root_path = cfg_data["root_path"]
    model_weights_path = os.path.join(cfg_data["output_dir"], "model_final.pth")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_data["model_config"]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_data["num_classes"]
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"

    return DefaultPredictor(cfg)

def visualize_predictions(predictor, cfg_data, num_samples=5):
    root_path = cfg_data["root_path"]
    val_images_path = os.path.join(root_path, cfg_data["val_images"])
    output_dir = os.path.join(cfg_data["output_dir"], "vis")
    os.makedirs(output_dir, exist_ok=True)

    val_images = [f for f in os.listdir(val_images_path) if f.lower().endswith((".jpg", ".png"))]
    sample_images = random.sample(val_images, min(num_samples, len(val_images)))

    for img_name in sample_images:
        img_path = os.path.join(val_images_path, img_name)
        img = cv2.imread(img_path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get("__unused").set(thing_classes=["Carton"]),
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_img = out.get_image()[:, :, ::-1]
        save_path = os.path.join(output_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, result_img)
        print(f"[✓] Saved: {save_path}")

def main():
    cfg_data = load_config()
    predictor = setup_predictor(cfg_data)
    visualize_predictions(predictor, cfg_data)

if __name__ == "__main__":
    main()
