# tach-phoma: YOLO26s-seg 1 lớp phoma

Dataset "BRACOL_SEGMENTATION (tách_phoma)": chỉ giữ nhãn phoma (id 2 -> 0), các lớp khác bỏ nhãn.

## Cấu hình
Giống `tach-lop/tach-miner/` và run 5, nc=1 (`data_phoma.yaml`):
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Không lặp ảnh, không trọng số lớp
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4
- Đánh giá thêm với conf >= 0.45, predict.py dùng conf 0.45

## Dataset (không đưa lên repo)
| Tập | Ảnh | Ảnh có phoma | Vết phoma |
|---|---|---|---|
| train | 1298 | 618 | 1450 |
| valid | 366 | 133 | 263 |
| test | 195 | 51 | 100 |

Để dataset ở `BRACOL_SEGMENTATION (tách_phoma)/`, chạy các script từ thư mục này.

## Chạy
```bash
python train.py
python resume.py   # nếu bị dừng giữa chừng
python evaluate.py
python predict.py path/to/images
```

## Kết quả
Đủ 150 epoch (bị dừng ở epoch 52 rồi train tiếp từ last.pt), best.pt là epoch 121.

| Tập | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Mask mAP50 (conf >= 0.45) |
|---|---|---|---|---|---|
| train | 0.741 | 0.806 | 0.840 | 0.616 | 0.712 |
| val | 0.709 | 0.693 | 0.754 | 0.553 | 0.668 |
| test | 0.740 | 0.653 | 0.702 | 0.539 | 0.647 |

### So với lớp phoma trong các bản 4 lớp (mask mAP50 / mAP50-95 / P)
| Tập | s3 (ưu tiên miner) | run 5 (không ưu tiên) | phoma 1 lớp |
|---|---|---|---|
| train | 0.893 / 0.667 / 0.83 | 0.891 / 0.677 / 0.77 | 0.840 / 0.616 / 0.74 |
| val | 0.763 / 0.565 / 0.61 | 0.753 / 0.574 / 0.48 | 0.754 / 0.553 / 0.71 |
| test | 0.733 / 0.583 / 0.55 | 0.709 / 0.565 / 0.37 | 0.702 / 0.539 / 0.74 |

Run 5 test trên bộ 180 ảnh, s3 và phoma 1 lớp test trên bộ 195 ảnh; cả hai đều có 100 vết phoma.

## Ghi chú
- mAP không tốt hơn bản 4 lớp.
- Precision cao hơn rõ (test 0.74 so với 0.37 - 0.55): ít báo nhầm phoma hơn, recall thấp hơn (0.65 so với 0.79 - 0.81).

Số liệu: `runs/y26s_seg_phoma_only/eval_summary.csv`.
