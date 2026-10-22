"""Danh gia best.pt tren train/val/test."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "BRACOL_SEGMENTATION (tách_miner)" / "data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="y26s_seg_miner_only")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    run = ROOT / "runs" / args.name
    model = YOLO(str(run / "weights" / "best.pt"))
    lines = ["split,images,instances,box_P,box_R,box_mAP50,box_mAP50-95,mask_P,mask_R,mask_mAP50,mask_mAP50-95"]
    for split in args.splits:
        r = model.val(data=str(DATA), split=split, imgsz=1024, batch=2, workers=0,
                      project=str(run), name=f"eval_{split}", exist_ok=True)
        lines.append(f"{split},{int(r.nt_per_image.sum())},{int(r.nt_per_class.sum())},"
                     + ",".join(f"{v:.4f}" for v in (*r.box.mean_results(), *r.seg.mean_results())))
    (run / "eval_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
