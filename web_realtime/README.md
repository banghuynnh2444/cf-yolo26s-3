# Web quét bệnh lá cà phê realtime

Ứng dụng web chạy model YOLO26s-seg (4 lớp: cercospora, miner, phoma, rust) theo thời gian thực.

- **Camera**: webcam laptop hoặc camera điện thoại, quét liên tục.
- **Video**: chọn file video, quét từng khung.
- **Ảnh**: tải lên (hoặc kéo thả) một hay nhiều ảnh, xem từng ảnh trong thư viện.

Hiển thị cho mỗi khung/ảnh:
- Mask phân đoạn, khung, nhãn và độ tin cậy của từng vết bệnh (rê chuột vào danh sách để làm nổi vết đó).
- Bảng theo bệnh: số vết, % diện tích khung hình, % diện tích lá, conf cao nhất và trung bình.
- Kết luận: bệnh chính, bệnh kèm theo, mức độ bệnh theo thang BRACOL (rất nhẹ ≤5%, nhẹ ≤10%, nặng ≤15%, rất nặng >15% diện tích lá).
- FPS, thời gian suy luận, độ trễ, biểu đồ số vết theo thời gian.
- Nhật ký các lần phát hiện, xuất CSV; lưu ảnh kết quả PNG.
- Chỉnh ngưỡng tin cậy từng lớp, imgsz, độ mờ mask, bật/tắt từng lớp; đổi qua lại giữa các best.pt có trong repo.

Model mặc định: `runs/y26s_seg_miner/weights/best.pt` (bản 3, mask mAP50 test 0.739, tốt nhất trong các lần train).
Ngưỡng mặc định giống `predict_new2.py`: phoma 0.45, các lớp khác 0.25.

## Cài đặt (Windows, dùng lại môi trường đã train)

```bat
cd D:\cf-yolo26s-3\web_realtime
pip install -r requirements.txt
```

`ultralytics` và `torch` bản CUDA đã có sẵn từ lúc train nên không cần cài lại.

## Chạy

```bat
python app.py
```
hoặc bấm đúp `run_web.bat`. Mở trình duyệt vào http://localhost:8000.

Tùy chọn:

| Lệnh | Ý nghĩa |
|---|---|
| `python app.py --model D:\duong\dan\best.pt` | dùng model khác |
| `python app.py --port 9000` | đổi cổng |
| `python app.py --device cpu` | chạy CPU |
| `python app.py --host 0.0.0.0 --https` | cho điện thoại cùng mạng Wi‑Fi truy cập và dùng camera |

### Dùng camera điện thoại
1. Chạy `python app.py --host 0.0.0.0 --https` (lần đầu tự tạo chứng chỉ `cert.pem`, `key.pem`).
2. Xem IP máy tính bằng `ipconfig` (ví dụ 192.168.1.10).
3. Trên điện thoại mở `https://192.168.1.10:8000`, chấp nhận cảnh báo chứng chỉ tự ký, chọn tab Camera, bấm "Bắt đầu quét". Trình duyệt chỉ cho dùng camera qua HTTPS hoặc localhost.
4. Nếu không vào được, cho phép Python qua Windows Firewall (mạng Private).

## Gợi ý
- Camera/video để imgsz 640 cho mượt; ảnh tĩnh để 1024 (đúng kích thước lúc train) cho chính xác nhất.
- Trên GPU, model chạy FP16 tự động. Thử trên CPU: imgsz 640 khoảng 110 ms/khung (~6 FPS); trên RTX 5050 sẽ nhanh hơn nhiều.
- "% lá" là ước lượng vùng lá theo màu (HSV), chính xác nhất khi chụp một lá trên nền sáng/đơn sắc như BRACOL. Ảnh ngoài vườn có nhiều lá làm nền thì chỉ nên xem số vết và % khung. Có thể tắt ở mục Cài đặt.
- Model còn yếu với cercospora (recall test ~0.46), nên kết quả cercospora cần kiểm tra lại bằng mắt.

## Cấu trúc
```
web_realtime/
  app.py            máy chủ FastAPI: /api/predict (ảnh), /ws (WebSocket cho camera/video), /api/model
  static/index.html giao diện
  static/app.js     gửi khung hình, vẽ mask, thống kê
  static/style.css
  run_web.bat
```

Luồng realtime: trình duyệt chụp khung hình (tối đa 1280px), nén JPEG, gửi qua WebSocket; máy chủ chạy YOLO, trả về polygon mask và thống kê dạng JSON; trình duyệt vẽ lên đúng khung đã gửi rồi mới gửi khung tiếp theo, nên không bị dồn hàng đợi.
