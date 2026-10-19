# cf_yolo26_5: YOLO26s-seg không ưu tiên lớp

4 lớp: cercospora, miner, phoma, rust. Ngưỡng tin cậy phoma 0.45 áp dụng khi đánh giá và dự đoán.

## Cấu hình
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Không lặp ảnh, không trọng số lớp (cls_pw 0)
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4

## Dataset (không đưa lên repo)
| Tập | Ảnh | cercospora | miner | phoma | rust |
|---|---|---|---|---|---|
| train | 1298 | 509 | 203 | 1450 | 4404 |
| valid | 366 | 105 | 118 | 263 | 1180 |
| test | 180 (chỉ ảnh BRACOL) | 42 | 72 | 100 | 707 |

## Chạy
```bash
python train.py
python resume.py   # nếu bị dừng giữa chừng
python evaluate.py
python make_report_tables.py
python predict.py path/to/images
```

## Kết quả
Đủ 150 epoch trong 4 giờ, best.pt là epoch 118.

Mask mAP50:

| Lớp | Train | Val | Test |
|---|---|---|---|
| all | 0.809 | 0.732 | 0.609 |
| cercospora | 0.808 | 0.638 | 0.175 |
| miner | 0.735 | 0.822 | 0.837 |
| phoma | 0.891 | 0.753 | 0.709 |
| rust | 0.804 | 0.715 | 0.715 |
| phoma (conf >= 0.45) | 0.764 | 0.651 | 0.657 |

### So sánh trên cùng tập test 180 ảnh BRACOL (mask mAP50)
| Lớp | s3 (ưu tiên miner) | 4 (ưu tiên cercospora) | 5 (không ưu tiên) |
|---|---|---|---|
| all | 0.638 | 0.615 | 0.609 |
| cercospora | 0.267 | 0.216 | 0.175 |
| miner | 0.840 | 0.827 | 0.837 |
| phoma | 0.733 | 0.701 | 0.709 |
| rust | 0.711 | 0.716 | 0.715 |
| mAP50-95 all | 0.440 | 0.424 | 0.429 |

s3 được đánh giá lại bằng best.pt của s3 trên tập test này.

## Ghi chú
- Cercospora trên ảnh BRACOL: recall 0.07, trong khi train 0.81 và val 0.64.
- Ngưỡng phoma 0.45 trên test: precision 0.37 -> 0.47, recall 0.81 -> 0.76, mAP50 0.709 -> 0.657.

Số liệu đầy đủ: `runs/y26s_seg/eval_summary.csv`, bảng `report_train/val/test.png`.
