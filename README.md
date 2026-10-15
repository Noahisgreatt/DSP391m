# VAD — Phát hiện lảng vảng & đi ngược dòng (DSP391m)

Pipeline phát hiện bất thường mức khung hình, bám theo luận án của NCS Trần Minh Tùng
(Spatial ViT → Temporal ViT → Temporal Cross-Attention → decoder dự đoán khung tương lai,
học không giám sát, đo bằng AUC/EER), và nâng cấp ở 3 điểm:

| Luận án | Dự án này |
|---|---|
| Dự đoán **khung RGB** tiếp theo | Dự đoán **optical flow** tiếp theo — lảng vảng/ngược dòng là bất thường về *chuyển động* |
| Giải mã từ một vector `[CLS]` | Giải mã từ **patch tokens** của khung cuối (giữ được *vị trí* của người bất thường) |
| Chỉ lỗi tái tạo | Thêm **đầu phân loại** học từ nhãn giả (YOLO+ByteTrack và flow) → biết *loại* bất thường |

Mô hình: `vit_flow` (ViT-B/16 pretrained ImageNet, ~128M tham số) và baseline `convlstm_flow`
(ResNet-50 + ConvLSTM, ~47M). `train.py` từ chối chạy nếu < `min_params` (35M).

## Cấu trúc
```
configs/features/{farneback,tracking}.yaml   # tham số đặc trưng / luật
configs/models/{vit_flow,convlstm_flow}.yaml # mỗi thí nghiệm một YAML
params.yaml                                  # đường dẫn dữ liệu, fps, chia tập
src/data/extract_frames.py   video/thư mục ảnh → data/interim/frames/<ds>/<split>__<id>/000001.jpg
src/data/dataset.py          cửa sổ L flow quá khứ → (flow kế tiếp, nhãn giả)
src/features/optical_flow.py Farneback → data/features/farneback/<ds>/<id>.npz (flow + hist)
src/features/direction.py    histogram hướng theo lưới, hướng chủ đạo mỗi ô của từng scene, nhãn ngược dòng từ flow
src/features/tracking.py     YOLO+ByteTrack → tracks + luật lảng vảng/ngược dòng → data/labels/tracking/
src/models/                  vit_flow.py, convlstm_flow.py (registry trong __init__.py)
src/train.py / evaluate.py / visualize.py
scripts/download_shanghaitech.py, download_gdrive.py
```
Mã video có tiền tố split (`train__01_001`, `test__01_0014`) nên mọi bước sau tự biết tập.
Nhãn lớp: `0` bình thường, `1` lảng vảng, `2` ngược dòng.

## Chạy
```bash
pip install -r requirements.txt
python scripts/download_shanghaitech.py            # hoặc --src /kaggle/input/... trên Kaggle
python scripts/download_gdrive.py --url <link thư mục> --out data/raw/manual/videos

dvc repro                                          # toàn bộ pipeline, hoặc từng bước:
python -m src.data.extract_frames --dataset shanghaitech
python -m src.features.optical_flow --dataset shanghaitech
python -m src.features.direction    --dataset shanghaitech
python -m src.features.tracking     --dataset shanghaitech     # cần GPU, chậm nhất
python -m src.evaluate --baseline tracking --dataset shanghaitech   # = cách làm trong notebook, trên toàn tập test
python -m src.train --config configs/models/vit_flow.yaml
python -m src.train --config configs/models/vit_flow.yaml --set train.lr=5e-5 train.epochs=10   # đổi tham số không sửa file
python -m src.evaluate --model vit_flow --dataset shanghaitech
python -m src.visualize --model vit_flow --dataset manual --all     # outputs/vit_flow/<run_id>/<video>.mp4
```
Mỗi lần train tạo `models/<model>/<run_id>/` (mặc định `YYYY-MM-DD_lr1e-4`) gồm
`config.yaml`, `best.pt`, `last.pt`, `metrics.json`; `latest.txt` trỏ tới lần chạy mới nhất.

## Những sửa so với notebook
- **Lảng vảng**: notebook đo từ *vị trí xuất hiện đầu tiên* với bán kính cố định 50 px → người đi vào rồi đứng lại không bao giờ bị bắt, người ở xa/gần bị đối xử khác nhau. Giờ: cửa sổ trượt `window_sec`, bán kính tính theo **chiều cao hộp**.
- **Ngược dòng**: notebook dùng tích vô hướng thô (`dot < -50`) → phụ thuộc tốc độ và độ phân giải. Giờ: **cosine** với hướng trung bình của đám đông (bỏ chính người đó), chỉ khi đám đông có hướng rõ, và phải kéo dài `min_frames`.
- **Đánh giá**: notebook tính AUC trên **một** clip (`01_0014`, nằm trong SHANGHAI_TRAIN). Giờ tính trên toàn bộ tập test, có EER (như luận án), AP, best-F1.
- Bản Kaggle của ShanghaiTech dùng cách chia weakly-supervised: một số clip *train* có bất thường. `params.yaml: train_with_anomaly: move_to_test` chuyển chúng sang test để huấn luyện không giám sát chỉ thấy dữ liệu bình thường.

## Lưu ý
- GT gốc của ShanghaiTech **không** coi đứng yên là bất thường; AUC trên ShanghaiTech đo khả năng phát hiện bất thường chung. Để đánh giá đúng hai hành vi mục tiêu, cần gán GT cho tập `data/raw/manual/` (`gt/<video_id>.npy`, mảng 0/1 mỗi khung — cùng định dạng UIT-ADrone).
- Dung lượng flow: 112×112 float16 ≈ 49 KB/khung (~13 GB cho train ShanghaiTech); tăng `frame_stride` nếu thiếu chỗ.
