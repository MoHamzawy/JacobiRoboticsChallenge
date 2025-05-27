import os
import yaml
import multiprocessing
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def load_config(path="segmentation/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
def save_cfg():
    cfg_dict = load_config()
    root = cfg_dict["root_path"]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_dict["model_config"]))
    cfg.DATASETS.TRAIN = ("carton_train",)
    cfg.DATASETS.TEST = ("carton_val",)
    cfg.DATALOADER.NUM_WORKERS = cfg_dict["num_workers"]
    cfg.SOLVER.IMS_PER_BATCH = cfg_dict["ims_per_batch"]
    cfg.SOLVER.BASE_LR = cfg_dict["base_lr"]
    cfg.SOLVER.MAX_ITER = cfg_dict["max_iter"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg_dict["batch_size_per_image"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_dict["num_classes"]
    cfg.OUTPUT_DIR = os.path.join(root, cfg_dict["output_dir"])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
        f.write(cfg.dump())
    print(f"✅ Saved cfg.yaml to: {cfg.OUTPUT_DIR}")

def main():
    cfg_dict = load_config()
    root = cfg_dict["root_path"]

    # Register COCO datasets
    register_coco_instances("carton_train", {}, os.path.join(root, cfg_dict["train_json"]), os.path.join(root, cfg_dict["train_images"]))
    register_coco_instances("carton_val", {}, os.path.join(root, cfg_dict["val_json"]), os.path.join(root, cfg_dict["val_images"]))

    # Detectron2 config setup
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_dict["model_config"]))
    cfg.DATASETS.TRAIN = ("carton_train",)
    cfg.DATASETS.TEST = ("carton_val",)
    cfg.DATALOADER.NUM_WORKERS = cfg_dict["num_workers"]
    cfg.SOLVER.IMS_PER_BATCH = cfg_dict["ims_per_batch"]
    cfg.SOLVER.BASE_LR = cfg_dict["base_lr"]
    cfg.SOLVER.MAX_ITER = cfg_dict["max_iter"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg_dict["batch_size_per_image"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_dict["num_classes"]
    cfg.OUTPUT_DIR = os.path.join(root, cfg_dict["output_dir"])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation after training
    evaluator = COCOEvaluator("carton_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"))
    val_loader = build_detection_test_loader(cfg, "carton_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    save_cfg()
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
