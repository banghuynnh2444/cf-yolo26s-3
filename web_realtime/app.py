"""Web quét bệnh lá cà phê realtime (YOLO26-seg).

Chạy:
    python app.py                       # mở http://localhost:8000
    python app.py --model path/to/best.pt --port 8000 --host 0.0.0.0
    python app.py --https               # bật HTTPS tự ký để điện thoại trong mạng LAN dùng camera

Mặc định dùng best.pt tốt nhất trong repo: runs/y26s_seg_miner/weights/best.pt (bản 3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DEFAULT_MODEL = REPO_DIR / "runs" / "y26s_seg_miner" / "weights" / "best.pt"

# Màu (BGR không dùng; frontend vẽ) và tên tiếng Việt cho 4 lớp
VI_NAMES = {
    "cercospora": "Đốm mắt cua (Cercospora)",
    "miner": "Sâu vẽ bùa (Leaf miner)",
    "phoma": "Thối đen lá (Phoma)",
    "rust": "Gỉ sắt (Rust)",
}
COLORS = {
    "cercospora": "#e8590c",
    "miner": "#7048e8",
    "phoma": "#1c7ed6",
    "rust": "#e03131",
}
# Ngưỡng mặc định theo predict_new2.py (phoma 0.45 để giảm báo nhầm)
DEFAULT_CONF = {"cercospora": 0.25, "miner": 0.25, "phoma": 0.45, "rust": 0.25}


def find_models() -> list[Path]:
    found = sorted(REPO_DIR.glob("**/weights/best.pt"))
    if DEFAULT_MODEL in found:
        found.remove(DEFAULT_MODEL)
        found.insert(0, DEFAULT_MODEL)
    return found


class Detector:
    def __init__(self, model_path: Path, device: str | None = None):
        self.lock = threading.Lock()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        self.load(model_path)

    def load(self, model_path: Path):
        model = YOLO(str(model_path))
        model.to(self.device)
        names = model.names
        self.names = [names[i] for i in range(len(names))]
        # Chạy nóng 1 lần để lần đầu không bị chậm
        model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, device=self.device,
                      half=self.half, verbose=False)
        with self.lock:
            self.model = model
            self.model_path = Path(model_path)

    def info(self) -> dict:
        gpu = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None
        try:
            rel = str(self.model_path.relative_to(REPO_DIR))
        except ValueError:
            rel = str(self.model_path)
        return {
            "model": rel,
            "models": [str(p.relative_to(REPO_DIR)) for p in find_models()],
            "device": self.device,
            "gpu": gpu,
            "half": self.half,
            "classes": [
                {"id": i, "name": n, "vi": VI_NAMES.get(n, n), "color": COLORS.get(n, "#888"),
                 "conf": DEFAULT_CONF.get(n, 0.25)}
                for i, n in enumerate(self.names)
            ],
        }

    def predict(self, img: np.ndarray, imgsz: int, conf: dict[str, float], leaf: bool) -> dict:
        h, w = img.shape[:2]
        min_conf = max(0.01, min(conf.get(n, 0.25) for n in self.names))
        t0 = time.perf_counter()
        with self.lock:
            r = self.model.predict(img, imgsz=imgsz, conf=min_conf, device=self.device,
                                   half=self.half, verbose=False, retina_masks=False)[0]
        t1 = time.perf_counter()

        dets = []
        n_cls = len(self.names)
        cls_masks = np.zeros((n_cls, h, w), np.uint8)
        if r.boxes is not None and len(r.boxes):
            cls_ids = r.boxes.cls.int().tolist()
            scores = r.boxes.conf.tolist()
            boxes = r.boxes.xyxy.tolist()
            polys = r.masks.xy if r.masks is not None else [np.zeros((0, 2))] * len(cls_ids)
            for c, s, b, p in zip(cls_ids, scores, boxes, polys):
                name = self.names[c]
                if s < conf.get(name, 0.25):
                    continue
                area = 0
                if len(p) >= 3:
                    pts = np.round(p).astype(np.int32)
                    area = int(round(cv2.contourArea(pts)))
                    cv2.fillPoly(cls_masks[c], [pts], 1)
                dets.append({
                    "cls": c,
                    "name": name,
                    "conf": round(float(s), 4),
                    "box": [round(v, 1) for v in b],
                    "poly": np.round(p, 1).tolist() if len(p) >= 3 else [],
                    "area_px": area,
                })
        t2 = time.perf_counter()

        total_px = h * w
        lesion_union = cls_masks.max(axis=0) if n_cls else np.zeros((h, w), np.uint8)
        leaf_px = None
        if leaf:
            leaf_px = int(estimate_leaf_mask(img, lesion_union).sum())

        summary = []
        for c, name in enumerate(self.names):
            ds = [d for d in dets if d["cls"] == c]
            px = int(cls_masks[c].sum())
            summary.append({
                "cls": c,
                "name": name,
                "count": len(ds),
                "area_px": px,
                "pct_frame": round(100 * px / total_px, 3),
                "pct_leaf": round(100 * px / leaf_px, 3) if leaf_px else None,
                "max_conf": round(max((d["conf"] for d in ds), default=0), 4),
                "mean_conf": round(float(np.mean([d["conf"] for d in ds])), 4) if ds else 0,
            })
        lesion_px = int(lesion_union.sum())
        return {
            "width": w,
            "height": h,
            "imgsz": imgsz,
            "infer_ms": round((t1 - t0) * 1000, 1),
            "post_ms": round((t2 - t1) * 1000, 1),
            "detections": dets,
            "summary": summary,
            "total": len(dets),
            "lesion_px": lesion_px,
            "lesion_pct_frame": round(100 * lesion_px / total_px, 3),
            "leaf_px": leaf_px,
            "leaf_pct_frame": round(100 * leaf_px / total_px, 2) if leaf_px else None,
            "lesion_pct_leaf": round(100 * lesion_px / leaf_px, 3) if leaf_px else None,
        }


def estimate_leaf_mask(img: np.ndarray, lesions: np.ndarray) -> np.ndarray:
    """Ước lượng vùng lá bằng màu (HSV): phần có màu lá (xanh/vàng/nâu) và đủ bão hòa.
    Chạy tốt với ảnh chụp lá trên nền sáng/đơn sắc như BRACOL; ảnh ngoài vườn (nền nhiều lá)
    thì con số này chỉ mang tính tham khảo."""
    small = cv2.resize(img, (img.shape[1] // 4 or 1, img.shape[0] // 4 or 1), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((hch >= 8) & (hch <= 95) & (s >= 45) & (v >= 35)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    # lấp lỗ bên trong lá (vết bệnh, đốm sáng)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.01 * m.size
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c for c in cnts if cv2.contourArea(c) >= min_area], -1, 1, cv2.FILLED)
    full = cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.maximum(full, lesions)


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def parse_conf(raw: str | dict | None) -> dict[str, float]:
    conf = dict(DEFAULT_CONF)
    if raw:
        d = json.loads(raw) if isinstance(raw, str) else raw
        conf.update({k: float(v) for k, v in d.items()})
    return conf


def create_app(detector: Detector) -> FastAPI:
    app = FastAPI(title="Quét bệnh lá cà phê")
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/api/info")
    def info():
        return detector.info()

    @app.post("/api/model")
    async def switch_model(path: str = Form(...)):
        p = (REPO_DIR / path).resolve()
        if p not in [m.resolve() for m in find_models()]:
            return JSONResponse({"error": "Không tìm thấy model"}, status_code=400)
        await asyncio.to_thread(detector.load, p)
        return detector.info()

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...), imgsz: int = Form(1024),
                      conf: str = Form(""), leaf: bool = Form(True)):
        img = decode_image(await file.read())
        if img is None:
            return JSONResponse({"error": "Không đọc được ảnh"}, status_code=400)
        res = await asyncio.to_thread(detector.predict, img, imgsz, parse_conf(conf), leaf)
        res["filename"] = file.filename
        return res

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        """Client gửi: text JSON {imgsz, conf, leaf} để đổi cấu hình, hoặc bytes = 1 khung JPEG.
        Server trả về JSON kết quả cho từng khung. Client chỉ gửi khung mới khi nhận kết quả cũ."""
        await sock.accept()
        cfg = {"imgsz": 640, "conf": dict(DEFAULT_CONF), "leaf": True}
        try:
            while True:
                msg = await sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text"):
                    d = json.loads(msg["text"])
                    if "imgsz" in d:
                        cfg["imgsz"] = int(d["imgsz"])
                    if "conf" in d:
                        cfg["conf"] = parse_conf(d["conf"])
                    if "leaf" in d:
                        cfg["leaf"] = bool(d["leaf"])
                    continue
                data = msg.get("bytes")
                if not data:
                    continue
                img = decode_image(data)
                if img is None:
                    await sock.send_json({"error": "Khung hình lỗi"})
                    continue
                res = await asyncio.to_thread(detector.predict, img, cfg["imgsz"], cfg["conf"], cfg["leaf"])
                await sock.send_json(res)
        except WebSocketDisconnect:
            pass

    return app


def make_self_signed(cert: Path, key: Path):
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "quet-benh-ca-phe.local")])
    now = datetime.now(timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now)
         .not_valid_after(now + timedelta(days=825)).sign(k, hashes.SHA256()))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="đường dẫn best.pt")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 để máy khác trong mạng LAN truy cập")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda:0 hoặc cpu (mặc định tự chọn)")
    ap.add_argument("--https", action="store_true", help="bật HTTPS tự ký (cần cho camera trên điện thoại)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Không thấy model: {model_path}")
    det = Detector(model_path, args.device)
    info = det.info()
    print(f"Model: {info['model']} | thiết bị: {info['device']} {info['gpu'] or ''} | lớp: {det.names}")

    ssl = {}
    scheme = "http"
    if args.https:
        cert, key = APP_DIR / "cert.pem", APP_DIR / "key.pem"
        if not cert.exists():
            make_self_signed(cert, key)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Mở trình duyệt: {scheme}://{shown}:{args.port}")
    uvicorn.run(create_app(det), host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
