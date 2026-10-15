"""Danh gia best.pt tren train/val/test, them che do phoma conf >= 0.45."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "BRACOL_SEGMENTATION" / "data.yaml"
PHOMA_CLS, PHOMA_CONF = 2, 0.45


class PhomaConfValidator(SegmentationValidator):
    def postprocess(self, preds):
        preds = super().postprocess(preds)
        for pred in preds:
            keep = ~((pred["cls"] == PHOMA_CLS) & (pred["conf"] < PHOMA_CONF))
            for k, v in pred.items():
                if hasattr(v, "shape") and v.shape[:1] == keep.shape:
                    pred[k] = v[keep]
        return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="y26s_seg_cercospora")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    run = ROOT / "runs" / args.name
    model = YOLO(str(run / "weights" / "best.pt"))
    lines = ["mode,split,class,images,instances,box_P,box_R,box_mAP50,box_mAP50-95,mask_P,mask_R,mask_mAP50,mask_mAP50-95"]
    for mode, validator in [("standard", None), ("phoma_conf_0.45", PhomaConfValidator)]:
        for split in args.splits:
            suffix = "" if validator is None else "_phoma045"
            kw = dict(data=str(DATA), split=split, imgsz=1024, batch=2, workers=0,
                      project=str(run), name=f"eval_{split}{suffix}", exist_ok=True)
            r = model.val(validator=validator, **kw) if validator else model.val(**kw)
            nt, ni = r.nt_per_class, r.nt_per_image
            lines.append(f"{mode},{split},all,,{int(nt.sum())},"
                         + ",".join(f"{v:.4f}" for v in (*r.box.mean_results(), *r.seg.mean_results())))
            for i, c in enumerate(r.ap_class_index):
                vals = (*r.box.class_result(i), *r.seg.class_result(i))
                lines.append(f"{mode},{split},{model.names[int(c)]},{int(ni[c])},{int(nt[c])},"
                             + ",".join(f"{v:.4f}" for v in vals))
    (run / "eval_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
