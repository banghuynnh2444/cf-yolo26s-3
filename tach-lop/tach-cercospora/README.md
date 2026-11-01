# tach-cercospora: YOLO26s-seg 1 lớp cercospora

Dataset "BRACOL_SEGMENTATION (tách_cerco)": chỉ giữ nhãn cercospora, các lớp khác bỏ nhãn.

## Cấu hình
Giống `tach-lop/tach-miner/` và run 5, nc=1:
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Không lặp ảnh, không trọng số lớp
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4

## Dataset (không đưa lên repo)
| Tập | Ảnh | Ảnh có cercospora | Vết cercospora |
|---|---|---|---|
| train | 1298 | 161 | 509 |
| valid | 366 | 43 | 105 |
| test | 195 | 40 | 91 |

Để dataset ở `BRACOL_SEGMENTATION (tách_cerco)/`.

## Chạy
```bash
python train.py
python resume.py   # nếu bị dừng giữa chừng
python evaluate.py
python predict.py path/to/images
```

## Kết quả
Đủ 150 epoch trong 5.2 giờ, best.pt là epoch 118.

| Tập | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| train | 0.855 | 0.625 | 0.784 | 0.450 |
| val | 0.813 | 0.498 | 0.614 | 0.338 |
| test | 0.638 | 0.522 | 0.610 | 0.306 |

### Tập test theo nguồn ảnh (mask)
| Nguồn ảnh | Ảnh | Vết | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| BRACOL gốc | 180 | 42 | 0.173 | 0.547 | 0.234 | 0.111 |
| Nguồn mới | 15 | 49 | 0.791 | 0.816 | 0.913 | 0.468 |

### So với lớp cercospora trong các bản khác (mask mAP50 / mAP50-95)
| Tập | s3 | run 4 | run 5 | cercospora 1 lớp |
|---|---|---|---|---|
| train | 0.813 / 0.495 | - | 0.808 / 0.501 | 0.784 / 0.450 |
| val 366 ảnh | 0.638 / 0.401 | - | 0.638 / 0.388 | 0.614 / 0.338 |
| test 180 ảnh BRACOL | 0.267 / 0.114 | 0.216 / 0.107 | 0.175 / 0.081 | 0.234 / 0.111 |

Run 4 dùng tập train và val khác nên chỉ so trên test.

## Ghi chú
- Tách riêng không cải thiện cercospora trên ảnh BRACOL.
- Trên ảnh BRACOL, recall 0.55 nhưng precision chỉ 0.17 (ở ngưỡng conf tối ưu F1).
- 15 ảnh nguồn mới có thể gần giống ảnh train, chưa loại trừ ảnh gần trùng.

Số liệu: `runs/y26s_seg_cercospora_only/eval_summary.csv`.
