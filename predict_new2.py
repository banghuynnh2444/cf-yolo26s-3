"""python predict_new2.py path/to/image_or_folder"""

import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "y26s_seg_miner"
CONF = {0: 0.25, 1: 0.25, 2: 0.45, 3: 0.25}  # cercospora, miner, phoma, rust

if __name__ == "__main__":
    model = YOLO(str(RUN / "weights" / "best.pt"))
    out = RUN / "predict"
    out.mkdir(parents=True, exist_ok=True)
    for r in model.predict(sys.argv[1], imgsz=1024, conf=min(CONF.values()), stream=True, verbose=False):
        keep = [i for i, (c, s) in enumerate(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist())) if s >= CONF[c]]
        r = r[keep]
        r.save(filename=str(out / Path(r.path).name))
        counts = r.boxes.cls.int().tolist()
        print(Path(r.path).name, {model.names[c]: counts.count(c) for c in CONF})
