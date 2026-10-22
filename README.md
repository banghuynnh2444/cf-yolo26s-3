# cf-yolo26s-3

YOLO26s-seg phân đoạn bệnh lá cà phê, 4 lớp: cercospora, miner, phoma, rust.
Cấu hình giống hệt run new2 (https://github.com/banghuynnh2444/cf_yolo26_new2), chỉ khác tập valid (366 ảnh, thêm 55 ảnh nền).

## Cấu hình
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Ảnh có miner lặp 3 lần trong tập train (1298 -> 1524)
- cls_pw 0.5, trọng số miner x1.5. Trọng số lớp: cercospora 1.216, miner 1.69, phoma 0.689, rust 0.406
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4
- GPU RTX 5050 Laptop 8 GB, workers 1, cache False

## Dataset (không đưa lên repo)
| Tập | Ảnh | cercospora | miner | phoma | rust |
|---|---|---|---|---|---|
| train | 1298 | 509 | 203 | 1450 | 4404 |
| valid | 366 | 105 | 118 | 263 | 1180 |
| test | 195 | 91 | 72 | 100 | 707 |

Để dataset ở `BRACOL_SEGMENTATION/`.

## Chạy
```bash
python train_miner.py
python resume_miner.py        # nếu bị dừng giữa chừng
python evaluate_new2.py
python make_report_tables.py
python predict_new2.py path/to/images
```

## Kết quả
best.pt là epoch 128, train đủ 150 epoch (khoảng 8.8 giờ).

Mask mAP50:

| Lớp | Train | Val | Test |
|---|---|---|---|
| all | 0.838 | 0.735 | 0.739 |
| cercospora | 0.813 | 0.638 | 0.671 |
| miner | 0.872 | 0.844 | 0.839 |
| phoma | 0.893 | 0.763 | 0.733 |
| rust | 0.774 | 0.697 | 0.711 |
| phoma (conf >= 0.45) | 0.745 | 0.636 | 0.677 |

Mask mAP50-95: train 0.574, val 0.507, test 0.501.

So với new2 trên test (mask mAP50): all 0.723 -> 0.739, cercospora 0.646 -> 0.671, miner 0.823 -> 0.839, phoma 0.715 -> 0.733, rust 0.707 -> 0.711.

Số liệu đầy đủ: `runs/y26s_seg_miner/eval_summary.csv`, bảng `report_train/val/test.png`.

## Ghi chú
- Cercospora recall trên test thấp (~0.46).
- Phoma precision trên test ~0.55. Ngưỡng 0.45 tăng precision lên ~0.60 nhưng mAP giảm.
- Rust mask mAP50-95 ~0.35, vết bệnh nhỏ.

## Run khác trong repo
- `cf_yolo26-4/`: YOLO26s-seg ưu tiên cercospora trên BRACOL gốc.
- `cf_yolo26_5/`: YOLO26s-seg không ưu tiên lớp, ngưỡng phoma 0.45.
- `tach-lop/tach-miner/`: YOLO26s-seg 1 lớp miner.
