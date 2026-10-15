"""Train YOLO26s-seg, uu tien lop cercospora (oversample + class weight)."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "BRACOL_SEGMENTATION"
WORK_DIR = ROOT / "cercospora_run_data"
NAMES = ["cercospora", "miner", "phoma", "rust"]
TARGET_CLS = 0  # cercospora


class CercosporaTrainer(SegmentationTrainer):
    target_boost = 1.5

    def compute_class_weights(self, class_counts):
        weights = super().compute_class_weights(class_counts)
        weights[TARGET_CLS] *= self.target_boost
        return weights


def build_oversampled_data_yaml(repeat: int) -> Path:
    WORK_DIR.mkdir(exist_ok=True)
    img_dir, lbl_dir = DATA_DIR / "train" / "images", DATA_DIR / "train" / "labels"
    lines, n_target = [], 0
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        has_target = lbl.exists() and any(
            ln.split()[0] == str(TARGET_CLS) for ln in lbl.read_text().splitlines() if ln.strip()
        )
        n_target += has_target
        lines += [img.as_posix()] * (repeat if has_target else 1)
    train_list = WORK_DIR / "train_oversampled.txt"
    train_list.write_text("\n".join(lines) + "\n")
    print(f"Train list: {len(lines)} entries ({n_target} cercospora images x{repeat})")

    data = {
        "path": DATA_DIR.as_posix(),
        "train": train_list.as_posix(),
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NAMES),
        "names": NAMES,
    }
    out = WORK_DIR / "data_oversampled.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26s-seg.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--cls_pw", type=float, default=0.5)
    p.add_argument("--boost", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", default="y26s_seg_cercospora")
    args = p.parse_args()

    CercosporaTrainer.target_boost = args.boost
    data_yaml = build_oversampled_data_yaml(args.repeat)
    model = YOLO(args.model)
    model.train(
        trainer=CercosporaTrainer,
        data=str(data_yaml),
        task="segment",
        epochs=args.epochs,
        patience=40,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=True,
        cache=False,
        workers=args.workers,
        device=0,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=15,
        cls_pw=args.cls_pw,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        project=str(ROOT / "runs"),
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    main()
