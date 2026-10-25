# tach-rust: YOLO26s-seg 1 lớp rust

Dataset "BRACOL_SEGMENTATION (tách_rust)": chỉ giữ nhãn rust, các lớp khác bỏ nhãn.

## Cấu hình
Giống `tach-lop/tach-miner/` và run 5, nc=1:
- yolo26s-seg, imgsz 1024, batch 4, 150 epoch, patience 40, AMP, cos_lr, close_mosaic 15
- Không lặp ảnh, không trọng số lớp
- Augment: mosaic 1.0, copy_paste 0.3, fliplr/flipud 0.5, degrees 10, hsv 0.015/0.6/0.4

## Dataset (không đưa lên repo)
| Tập | Ảnh | Ảnh có rust | Vết rust |
|---|---|---|---|
| train | 1298 | 466 | 4404 |
| valid | 366 | 148 | 1180 |
| test | 195 | 96 | 707 |

Để dataset ở `BRACOL_SEGMENTATION (tách_rust)/`.

## Chạy
```bash
python train.py
python resume.py   # nếu bị dừng giữa chừng
python evaluate.py
python predict.py path/to/images
```

## Kết quả
Đủ 150 epoch trong 4.7 giờ, best.pt là epoch 135.

| Tập | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| train | 0.713 | 0.728 | 0.764 | 0.382 |
| val | 0.674 | 0.681 | 0.708 | 0.351 |
| test | 0.684 | 0.680 | 0.698 | 0.338 |

### So với lớp rust trong các bản 4 lớp (mask mAP50 / mAP50-95)
| Tập | s3 (ưu tiên miner) | run 5 (không ưu tiên) | rust 1 lớp |
|---|---|---|---|
| train | 0.774 / 0.393 | 0.804 / 0.421 | 0.764 / 0.382 |
| val | 0.697 / 0.359 | 0.715 / 0.369 | 0.708 / 0.351 |
| test | 0.711 / 0.353 | 0.715 / 0.359 | 0.698 / 0.338 |

Run 5 test trên bộ 180 ảnh, s3 và rust 1 lớp test trên bộ 195 ảnh; cả hai đều có 707 vết rust.

## Ghi chú
- Tách riêng rust không tốt hơn bản 4 lớp, thấp hơn khoảng 0.01 - 0.02.
- mAP50-95 thấp (~0.34 - 0.35) ở mọi bản: vết rust rất nhỏ ở imgsz 1024.

Số liệu: `runs/y26s_seg_rust_only/eval_summary.csv`.
