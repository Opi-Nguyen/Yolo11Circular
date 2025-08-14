import os
import yaml
from dataloader.CircleYOLOv11Dataset import CircleYOLOv11Dataset

def load_circle_yolo11_datasets(yaml_path, transform=None, img_size=448):
    """
    Tải các tập train/val/test cho CircleYOLO từ file dataset.yaml theo chuẩn YOLOv11.
    Tự động chuẩn hóa đường dẫn tránh lỗi FileNotFound.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", ".")
    # Chuẩn hóa đường dẫn bằng cách join và norm
    def norm(p): return os.path.normpath(os.path.join(root, p))

    train_img_dir = norm(cfg["train"])
    val_img_dir   = norm(cfg["val"])
    test_img_dir  = norm(cfg["test"])

    train_lbl_dir = train_img_dir.replace("images", "labels")
    val_lbl_dir   = val_img_dir.replace("images", "labels")
    test_lbl_dir  = test_img_dir.replace("images", "labels")

    dataset_train = CircleYOLOv11Dataset(train_img_dir, train_lbl_dir, img_size=img_size, transform=transform)
    dataset_val   = CircleYOLOv11Dataset(val_img_dir,   val_lbl_dir,   img_size=img_size, transform=transform)
    dataset_test  = CircleYOLOv11Dataset(test_img_dir,  test_lbl_dir,  img_size=img_size, transform=transform)

    return dataset_train, dataset_val, dataset_test
