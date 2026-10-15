"""Train tiep tu last.pt."""
from ultralytics import YOLO
from train_miner import ROOT, MinerTrainer

if __name__ == "__main__":
    YOLO(str(ROOT / "runs" / "y26s_seg_miner" / "weights" / "last.pt")).train(resume=True, trainer=MinerTrainer)
