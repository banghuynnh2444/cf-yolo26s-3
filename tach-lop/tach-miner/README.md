# tach-miner: YOLO26s-seg 1 lớp miner

Dataset "BRACOL_SEGMENTATION (tách_miner)": chỉ giữ nhãn miner, các lớp khác bỏ nhãn.

## Cấu hình
Giống run 5 (`cf_yolo26_5/`), nc=1:
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Không lặp ảnh, không trọng số lớp
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4

## Dataset (không đưa lên repo)
| Tập | Ảnh | Ảnh có miner | Vết miner |
|---|---|---|---|
| train | 1298 | 113 | 203 |
| valid | 366 | 69 | 118 |
| test | 195 | 41 | 72 |

Để dataset ở `BRACOL_SEGMENTATION (tách_miner)/`.

## Chạy
```bash
python train.py
python resume.py   # nếu bị dừng giữa chừng
python evaluate.py
python predict.py path/to/images
```

## Kết quả
Đủ 150 epoch trong 3.2 giờ, best.pt là epoch 115.

| Tập | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| train | 0.660 | 0.651 | 0.693 | 0.578 |
| val | 0.838 | 0.700 | 0.805 | 0.663 |
| test | 0.865 | 0.621 | 0.819 | 0.664 |

### So với lớp miner trong các bản 4 lớp (mask mAP50 / mAP50-95)
| Tập | s3 (ưu tiên miner) | run 5 (không ưu tiên) | miner 1 lớp |
|---|---|---|---|
| train | 0.872 / 0.741 | 0.735 / 0.624 | 0.693 / 0.578 |
| val | 0.844 / 0.702 | 0.822 / 0.678 | 0.805 / 0.663 |
| test | 0.839 / 0.708 | 0.837 / 0.710 | 0.819 / 0.664 |

Run 5 test trên bộ 180 ảnh, s3 và miner 1 lớp test trên bộ 195 ảnh; cả hai đều có 72 vết miner.

## Ghi chú
- Tách riêng miner không tốt hơn bản 4 lớp.
- Precision cao (0.84 - 0.87) nhưng recall thấp (0.62 - 0.70).
- Điểm trên train thấp hơn val và test, chưa rõ nguyên nhân.

Số liệu: `runs/y26s_seg_miner_only/eval_summary.csv`.
