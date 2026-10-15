"""Ve bang ket qua report_<split>.png tu eval_summary.csv."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "y26s_seg_miner"
COLS = ["Class", "Images", "Instances", "Box P", "Box R", "Box mAP50", "Box mAP50-95",
        "Mask P", "Mask R", "Mask mAP50", "Mask mAP50-95"]
METRICS = ["box_P", "box_R", "box_mAP50", "box_mAP50-95", "mask_P", "mask_R", "mask_mAP50", "mask_mAP50-95"]
N_IMAGES = {"train": len(list((ROOT / "BRACOL_SEGMENTATION/train/images").glob("*.jpg"))),
            "val": len(list((ROOT / "BRACOL_SEGMENTATION/valid/images").glob("*.jpg"))),
            "test": len(list((ROOT / "BRACOL_SEGMENTATION/test/images").glob("*.jpg")))}


def render(split, rows):
    fig, ax = plt.subplots(figsize=(18, 3.6))
    ax.axis("off")
    ax.set_title(f"YOLO26s-seg - best.pt epoch 128 - {split.upper()} "
                 f"({N_IMAGES[split]} images, imgsz 1024)", fontsize=14, fontweight="bold", pad=12)
    t = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.8)
    t.auto_set_column_width(list(range(len(COLS))))
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f4f6f")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        name = rows[r - 1][0]
        if name == "all":
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(fontweight="bold")
        elif name == "miner":
            cell.set_facecolor("#e2f0e2")
        elif name.startswith("phoma (conf"):
            cell.set_facecolor("#fff4d6")
        if c in (9, 10):
            cell.set_text_props(fontweight="bold")
    out = RUN / f"report_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    data = list(csv.DictReader(open(RUN / "eval_summary.csv")))
    for split in ["train", "val", "test"]:
        rows = []
        for d in data:
            if d["split"] != split:
                continue
            if d["mode"] == "standard":
                images = N_IMAGES[split] if d["class"] == "all" else d["images"]
                rows.append([d["class"], images, d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
            elif d["class"] == "phoma":
                rows.append(["phoma (conf>=0.45)", d["images"], d["instances"], *[f"{float(d[m]):.3f}" for m in METRICS]])
        render(split, rows)
