"""Train YOLO26s-seg, 1 lop miner."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "BRACOL_SEGMENTATION (tách_miner)" / "data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26s-seg.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--name", default="y26s_seg_miner_only")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        task="segment",
        epochs=args.epochs,
        patience=40,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=True,
        cache=False,  # cache="ram" bi loi tren Windows
        workers=args.workers,
        device=0,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=15,
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
