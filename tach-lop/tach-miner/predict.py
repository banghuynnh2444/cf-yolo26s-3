"""python predict.py path/to/image_or_folder"""

import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "y26s_seg_miner_only"

if __name__ == "__main__":
    model = YOLO(str(RUN / "weights" / "best.pt"))
    out = RUN / "predict"
    out.mkdir(parents=True, exist_ok=True)
    for r in model.predict(sys.argv[1], imgsz=1024, conf=0.25, stream=True, verbose=False):
        r.save(filename=str(out / Path(r.path).name))
        print(Path(r.path).name, len(r.boxes))
