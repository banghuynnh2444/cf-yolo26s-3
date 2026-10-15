"""Train tiep tu last.pt."""
from ultralytics import YOLO
from train_cercospora import ROOT, CercosporaTrainer

if __name__ == "__main__":
    YOLO(str(ROOT / "runs" / "y26s_seg_cercospora" / "weights" / "last.pt")).train(resume=True, trainer=CercosporaTrainer)
