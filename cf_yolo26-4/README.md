# cf_yolo26-4: YOLO26s-seg ưu tiên cercospora

Dataset BRACOL gốc (không có ảnh nguồn mới), 4 lớp: cercospora, miner, phoma, rust.

## Cấu hình
Giống run s3 (thư mục gốc repo), chỉ đổi lớp ưu tiên từ miner sang cercospora:
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Ảnh có cercospora lặp 3 lần trong tập train (1213 -> 1365)
- cls_pw 0.5, trọng số cercospora x1.5. Trọng số lớp: cercospora 1.702, miner 1.448, phoma 0.546, rust 0.304
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4

## Dataset (không đưa lên repo)
| Tập | Ảnh | cercospora | miner | phoma | rust |
|---|---|---|---|---|---|
| train | 1213 | 120 | 203 | 1450 | 4404 |
| valid | 351 | 43 | 118 | 263 | 1180 |
| test | 180 | 42 | 72 | 100 | 707 |

## Chạy
```bash
python train_cercospora.py
python resume_cercospora.py   # nếu bị dừng giữa chừng
python evaluate.py
python make_report_tables.py
python predict.py path/to/images
```

## Kết quả
Dừng sớm ở epoch 110 (patience 40), best.pt là epoch 70, khoảng 6.2 giờ.

Mask mAP50:

| Lớp | Train | Val | Test |
|---|---|---|---|
| all | 0.691 | 0.617 | 0.615 |
| cercospora | 0.425 | 0.190 | 0.216 |
| miner | 0.703 | 0.839 | 0.827 |
| phoma | 0.866 | 0.743 | 0.701 |
| rust | 0.770 | 0.697 | 0.716 |
| phoma (conf >= 0.45) | 0.696 | 0.613 | 0.620 |

Mask mAP50-95: train 0.468, val 0.426, test 0.424.

So sánh trên 180 ảnh BRACOL của tập test (mask mAP50): cercospora run 1 0.243, new2 0.220, run này 0.216. Tất cả: 0.628, 0.619, 0.615.

## Ghi chú
- Lặp ảnh và tăng trọng số không cải thiện cercospora.
- Cercospora trên tập train chỉ đạt 0.425, các lớp khác 0.70 - 0.87. Có thể do quá ít mẫu (120 vết) hoặc nhãn chưa đồng nhất, chưa kiểm chứng.

Số liệu đầy đủ: `runs/y26s_seg_cercospora/eval_summary.csv`, bảng `report_train/val/test.png`.
