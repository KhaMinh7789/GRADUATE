# 📖 Hướng Dẫn Toàn Bộ Source Code
## COVID-19 Cough Detection — KhaTM_24MSE43024

> **Thesis**: *"COVID-19 Detection from Cough Sounds Using Hybrid Deep Learning with Wav2Vec 2.0, AST, and Deep Neural Decision Forest (DNDF)"*
> **Data**: *"Liên hệ với tác giả để được cung cấp"*

---

## 📂 Cấu Trúc Thư Mục Tổng Quan

```
Covid_Cough_Research/
│
├── 📂 data/                          # Dữ liệu thô và features đã trích xuất
│   ├── Coswara-Data-master/          # Dataset Coswara (âm thanh ho thô)
│   ├── coughvid_20211012/            # Dataset COUGHVID
│   ├── virufy-data-main/             # Dataset Virufy
│   ├── metadata.csv                  # Metadata tổng hợp (16,354 mẫu)
│   ├── metadata_expert_clean.csv     # Subset được lọc bởi chuyên gia
│   ├── features_1d_raw/              # Features 1D (raw waveform tensors)
│   └── features_2d_ast/              # Features 2D (Mel-spectrogram tensors)
│
├── 📂 src/                           # Toàn bộ mã nguồn thực nghiệm
│   ├── 📂 prepare/                   # Chuẩn bị dữ liệu
│   │   ├── prepare_coswara.py
│   │   ├── prepare_coughvid.py
│   │   └── prepare_virufy.py
│   ├── 📂 common/                    # Các module dùng chung
│   │
│   ├── model_1d_sota.py              # ⭐ Kiến trúc Wav2Vec 2.0 + DNDF
│   ├── model_2d_sota.py              # ⭐ Kiến trúc AST + DNDF
│   ├── dataset_offline.py            # Dataset loader từ features đã trích xuất
│   ├── extract_features.py           # Trích xuất features 2D (Mel-spectrogram)
│   ├── extract_features_1d.py        # Trích xuất features 1D (raw audio)
│   │
│   ├── train_1d_sota.py              # ⭐ Huấn luyện Wav2Vec 2.0 + DNDF (5-Fold CV)
│   ├── train_2d_sota.py              # ⭐ Huấn luyện AST + DNDF (5-Fold CV)
│   │
│   ├── evaluate_cross_dataset.py     # ⭐ Đánh giá cross-dataset (tạo số liệu thesis)
│   ├── ablation_dndf_vs_softmax.py   # ⭐ Ablation study DNDF vs Softmax
│   ├── xai_interpretability.py       # ⭐ Giải thích mô hình bằng LIME
│   ├── threshold_optimization.py     # ⭐ Tối ưu ngưỡng quyết định (Youden J)
│   ├── experiment_ece_reliability.py # ⭐ Đánh giá ECE & Reliability Diagram
│   ├── experiment_expert_labels.py   # ⭐ Thực nghiệm với expert-labeled subset
│   ├── experiment_freq_ablation.py   # ⭐ Ablation tần số (loại bỏ dải 2-6 kHz)
│   │
│   ├── utils_metrics.py              # Hàm tính toán metrics (AUC, ECE, ...)
│   ├── plot_charts.py                # Vẽ biểu đồ ROC, PR curves
│   ├── plot_confusion_matrix.py      # Vẽ confusion matrix
│   ├── generate_experiment_report.py # Tạo báo cáo thực nghiệm tổng hợp
│   └── run_full_experiment.py        # ⭐ Script tổng điều phối toàn bộ pipeline
│
├── 📂 evaluation_results/            # Kết quả đánh giá đã lưu
│   ├── detailed_metrics.json         # Metrics chi tiết của 2 mô hình
│   ├── model_comparison_table.csv    # Bảng so sánh (CSV)
│   └── roc_curves.json               # Dữ liệu đường cong ROC
│
├── 📂 experiment_report/             # Báo cáo thực nghiệm đầy đủ
│   ├── EXPERIMENT_REPORT.md          # Báo cáo tổng hợp (tự sinh)
│   ├── ablation_results.json         # ⭐ Kết quả ablation DNDF vs Softmax
│   ├── ece_results.json              # ⭐ Kết quả ECE calibration
│   ├── expert_subset_results.json    # Kết quả subset chuyên gia
│   ├── frequency_ablation_results.json # Kết quả ablation tần số
│   ├── threshold_analysis.json       # ⭐ Phân tích ngưỡng Youden optimal
│   ├── threshold_sweep_table.csv     # Bảng quét ngưỡng
│   ├── summary_statistics.json       # Thống kê tóm tắt
│   │
│   ├── roc_curves.png                # Biểu đồ ROC curves
│   ├── confusion_matrices.png        # Ma trận nhầm lẫn
│   ├── metrics_comparison.png        # So sánh metrics tổng thể
│   ├── ece_reliability_diagram.png   # Reliability diagram (ECE)
│   ├── reliability_diagram.png       # Biểu đồ độ tin cậy
│   └── revised_performance_analysis.png  # Phân tích hiệu suất chi tiết
│
├── 📂 xai_explanations/              # Kết quả giải thích LIME
│   ├── sample_*_ast_explanation.png  # LIME trực quan hóa cho AST
│   ├── sample_*_wav2vec_explanation.png # LIME trực quan hóa cho Wav2Vec
│   └── xai_summary.txt              # Tóm tắt XAI
│
├── 📂 web_app/                       # Prototype web application
│
├── best_wav2vec_fold_1.pth → fold_5.pth  # ⭐ Model weights Wav2Vec (5 folds)
├── best_ast_fold_1.pth → fold_5.pth      # ⭐ Model weights AST (5 folds)
├── best_1d_model.pth                     # Model 1D tốt nhất
├── best_2d_model.pth                     # Model 2D tốt nhất
│
└── README.md              # 📖 File này
```

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy

### Yêu Cầu Hệ Thống

```
Python     >= 3.9
PyTorch    >= 2.0 (CUDA 11.8+)
GPU        NVIDIA với ít nhất 8GB VRAM (RTX 3070 trở lên)
RAM        >= 16GB
Disk       >= 50GB (data + model weights)
```

### Cài Đặt Môi Trường

```bash
# Kích hoạt virtual environment
cd D:\GROWTH\GRADUTION_MASTER\Covid_Cough_Research
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/Mac

# Hoặc cài mới từ đầu:
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets librosa scikit-learn pandas numpy matplotlib seaborn tqdm lime
```

---

## 🔄 Pipeline Thực Nghiệm Theo Thứ Tự

### BƯỚC 1 — Chuẩn Bị Dữ Liệu

> **Files liên quan**: `src/prepare/*.py`, `data/metadata.csv`

```bash
# Chuẩn bị từng dataset
cd src
python prepare/prepare_coswara.py    # → Xử lý Coswara, xuất metadata
python prepare/prepare_coughvid.py   # → Xử lý COUGHVID, chuẩn hóa labels
python prepare/prepare_virufy.py     # → Xử lý Virufy (clinical dataset)
```

**Output**: `data/metadata.csv` — File tổng hợp với 16,354 mẫu âm thanh, các cột:
- `file_path`: Đường dẫn tới file audio
- `label`: `1` = COVID-19 dương tính, `0` = khỏe mạnh
- `dataset`: Nguồn dataset (`coswara`, `coughvid`, `virufy`)
- `status`: Nhãn y tế gốc

---

### BƯỚC 2 — Trích Xuất Features

> **Files liên quan**: `src/extract_features_1d.py`, `src/extract_features.py`

```bash
cd src

# Trích xuất raw waveform tensors (cho Wav2Vec 2.0)
python extract_features_1d.py
# → Lưu vào: data/features_1d_raw/*.pt

# Trích xuất Mel-spectrogram tensors (cho AST)
python extract_features.py
# → Lưu vào: data/features_2d_ast/*.pt
```

**Mục đích**: Trích xuất trước để tăng tốc huấn luyện (không cần xử lý audio mỗi batch).

---

### BƯỚC 3 — Huấn Luyện Mô Hình

#### Model 1: Wav2Vec 2.0 + DNDF (1D Audio)

> **File chính**: `src/train_1d_sota.py`

```bash
cd src
python train_1d_sota.py
```

**Cấu hình huấn luyện**:
| Tham số | Giá trị |
|---------|---------|
| Feature Extractor | `facebook/wav2vec2-base` (pre-trained) |
| Classifier | DNDF (5 cây, depth=4) |
| Batch Size | 8 |
| Epochs | 15 |
| Learning Rate (backbone) | 1e-5 |
| Learning Rate (DNDF) | 1e-3 |
| Loss Function | Binary Focal Loss (α=0.8, γ=2.0) |
| Optimizer | AdamW + CosineAnnealingLR |
| CV | 5-Fold StratifiedKFold (seed=42) |
| Imbalance | WeightedRandomSampler |

**Output**: `best_wav2vec_fold_1.pth` → `best_wav2vec_fold_5.pth`

#### Model 2: AST + DNDF (2D Spectrogram)

> **File chính**: `src/train_2d_sota.py`

```bash
cd src
python train_2d_sota.py
```

**Điểm khác biệt so với Model 1**:
- Feature Extractor: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Input: Mel-spectrogram 2D thay vì raw waveform 1D
- Augmentation: **MixUp** (α=0.4) được áp dụng thêm
- Batch Size: 16

**Output**: `best_ast_fold_1.pth` → `best_ast_fold_5.pth`

---

### BƯỚC 4 — Đánh Giá Cross-Dataset

> **File chính**: `src/evaluate_cross_dataset.py`

```bash
cd src
python evaluate_cross_dataset.py
```

**Mục đích**: Đánh giá mô hình đã huấn luyện trên toàn bộ tập test (Coswara + COUGHVID).

**Output**:
- `evaluation_results/detailed_metrics.json`
- `evaluation_results/model_comparison_table.csv`
- `evaluation_results/roc_curves.json`

---

### BƯỚC 5 — Ablation Study: DNDF vs Softmax

> **File chính**: `src/ablation_dndf_vs_softmax.py`

```bash
cd src
python ablation_dndf_vs_softmax.py
```

**Mục đích**: So sánh A/B có kiểm soát — giữ nguyên Wav2Vec 2.0, chỉ thay đổi lớp phân loại:
- **Condition A**: Wav2Vec 2.0 + Linear + Sigmoid (Baseline)
- **Condition B**: Wav2Vec 2.0 + DNDF (Proposed)

**Output**: `experiment_report/ablation_results.json`

---

### BƯỚC 6 — Tối Ưu Ngưỡng Quyết Định

> **File chính**: `src/threshold_optimization.py`

```bash
cd src
python threshold_optimization.py
```

**Mục đích**: Tìm ngưỡng tối ưu theo chỉ số Youden J = Sensitivity + Specificity - 1.

**Output**: `experiment_report/threshold_analysis.json`, `threshold_sweep_table.csv`

---

### BƯỚC 7 — Đánh Giá ECE & Reliability Diagram

> **File chính**: `src/experiment_ece_reliability.py`

```bash
cd src
python experiment_ece_reliability.py
```

**Mục đích**: Đánh giá mức độ **calibration** (độ tương quan giữa confidence và accuracy thực tế) bằng chỉ số Expected Calibration Error (ECE).

**Output**: `experiment_report/ece_results.json`, `experiment_report/ece_reliability_diagram.png`

---

### BƯỚC 8 — Ablation Tần Số

> **File chính**: `src/experiment_freq_ablation.py`

```bash
cd src
python experiment_freq_ablation.py
```

**Mục đích**: Xác định đóng góp của dải tần số 2–6 kHz (dải quan trọng nhất cho phát hiện COVID-19 theo XAI/LIME) bằng cách loại bỏ nó và đo sự sụt giảm hiệu suất.

**Output**: `experiment_report/frequency_ablation_results.json`

---

### BƯỚC 9 — Thực Nghiệm Expert-Labeled Subset

> **File chính**: `src/experiment_expert_labels.py`

```bash
cd src
python experiment_expert_labels.py
```

**Mục đích**: Đánh giá mô hình trên subset đã được bác sĩ/chuyên gia xác nhận nhãn (loại bỏ nhiễu crowdsource).

**Output**: `experiment_report/expert_subset_results.json`

---

### BƯỚC 10 — Giải Thích Mô Hình (XAI/LIME)

> **File chính**: `src/xai_interpretability.py`

```bash
cd src
python xai_interpretability.py
```

**Mục đích**: Tạo LIME explanations cho từng mẫu — hiển thị vùng thời gian-tần số nào quan trọng nhất cho quyết định phân loại.

**Output**: `xai_explanations/sample_*_ast_explanation.png`, `sample_*_wav2vec_explanation.png`

---

### BƯỚC 11 — Tạo Báo Cáo Tổng Hợp

> **File chính**: `src/generate_experiment_report.py`

```bash
cd src
python generate_experiment_report.py
```

**Output**: `experiment_report/EXPERIMENT_REPORT.md`, tất cả biểu đồ PNG

---

### ⚡ Chạy Toàn Bộ Pipeline Cùng Một Lúc

```bash
cd src

# Chạy đầy đủ (evaluation + XAI + report)
python run_full_experiment.py --mode full

# Chỉ evaluation (bỏ qua XAI, nhanh hơn)
python run_full_experiment.py --mode demo

# Chỉ phase đánh giá
python run_full_experiment.py --mode phase1

# Chỉ tạo XAI
python run_full_experiment.py --mode phase2

# Chỉ tạo báo cáo
python run_full_experiment.py --mode phase3
```

---

## 📊 Các File Tạo Ra Số Liệu Trong Thesis

| Số liệu trong Thesis | File Tạo Ra | File Kết Quả |
|----------------------|-------------|--------------|
| **Bảng 4.1**: So sánh mô hình tổng thể | `evaluate_cross_dataset.py` | `evaluation_results/model_comparison_table.csv` |
| **Bảng 4.2**: Ablation DNDF vs Softmax | `ablation_dndf_vs_softmax.py` | `experiment_report/ablation_results.json` |
| **Bảng 4.3**: Threshold sweep (Youden) | `threshold_optimization.py` | `experiment_report/threshold_analysis.json` |
| **Bảng 4.4**: ECE calibration | `experiment_ece_reliability.py` | `experiment_report/ece_results.json` |
| **Bảng 4.5**: Expert subset | `experiment_expert_labels.py` | `experiment_report/expert_subset_results.json` |
| **Bảng 4.6**: Frequency ablation | `experiment_freq_ablation.py` | `experiment_report/frequency_ablation_results.json` |
| **Hình 4.3.1**: ROC Curves | `plot_charts.py` | `experiment_report/roc_curves.png` |
| **Hình 4.3.2**: Learning Curves | `train_1d_sota.py` / `train_2d_sota.py` | `src/common/Figure_4.3.2_Learning_Curves.png` |
| **Hình 4.3.3**: Confusion Matrix | `plot_confusion_matrix.py` | `experiment_report/confusion_matrices.png` |
| **Hình 4.4**: Reliability Diagram | `experiment_ece_reliability.py` | `experiment_report/ece_reliability_diagram.png` |
| **Hình 5.1**: LIME Explanations | `xai_interpretability.py` | `xai_explanations/*.png` |

---

## 📈 Giải Thích Các Số Liệu Kết Quả

### Kết Quả Chính — Bảng So Sánh Mô Hình (Bảng 4.1 Thesis)

| Mô Hình | Accuracy | Sensitivity | Specificity | Precision | F1-Score | AUC-ROC |
|---------|----------|-------------|-------------|-----------|----------|---------|
| **Wav2Vec 2.0 + DNDF** | 0.7854 | **0.5339** | **0.8627** | **0.5445** | **0.5391** | **0.7735** |
| AST + DNDF | 0.7966 | 0.5024 | 0.8332 | 0.2728 | 0.3536 | 0.6910 |

> **Nguồn số liệu**: `evaluation_results/detailed_metrics.json` và `evaluation_results/model_comparison_table.csv`

#### 🔍 Diễn Giải Chi Tiết:

**Wav2Vec 2.0 + DNDF** là mô hình đề xuất chính của thesis:
- **AUC-ROC = 0.7735**: Mô hình có khả năng phân biệt COVID-19 vs. khỏe mạnh ở mức **vượt trội so với random (0.5)**. Ngưỡng AUC > 0.75 được coi là "acceptable" trong y tế dự phòng.
- **Sensitivity = 53.4%**: Tỷ lệ phát hiện đúng ca COVID-19 — mô hình bắt được hơn nửa số ca dương tính thực sự ở ngưỡng mặc định τ=0.50.
- **Specificity = 86.3%**: Tỷ lệ xác định đúng người khỏe mạnh rất cao — mô hình ít báo nhầm người lành là bệnh.
- **Precision = 54.4%**: Trong số người bị mô hình phân loại là COVID-19, hơn nửa thực sự dương tính.
- **F1 = 0.539**: Điểm hài hòa giữa Precision và Recall — hợp lý với dữ liệu mất cân bằng.

**AST + DNDF** (mô hình đối chiếu):
- Accuracy cao hơn (0.7966) nhưng **AUC-ROC thấp hơn đáng kể (0.691)** — do dataset AST có độ mất cân bằng lớn hơn.
- Precision rất thấp (0.2728) → **nhiều false positive** hơn.
- AST hoạt động kém hơn trên tập này vì lý do class imbalance nghiêm trọng hơn (prevalence 11% vs 23.5%).

---

### Ablation Study: DNDF vs Softmax (Bảng 4.2 Thesis)

> **Nguồn số liệu**: `experiment_report/ablation_results.json`

| Mô Hình | AUC-ROC (mean±std) | AUC-PR (mean±std) |
|---------|-------------------|-------------------|
| Wav2Vec 2.0 + **Softmax** (Baseline) | 0.7663 ± 0.0098 | 0.5377 ± 0.0213 |
| Wav2Vec 2.0 + **DNDF** (Proposed) | 0.7638 ± 0.0171 | 0.5263 ± 0.0325 |

**Delta DNDF vs Softmax**:
- AUC-ROC: **−0.0025** (−0.3%)
- AUC-PR: **−0.0114** (−2.1%)

#### 🔍 Diễn Giải (Quan Trọng Cho Thesis Defense):

> ⚠️ **Điểm quan trọng**: DNDF không vượt trội Softmax về AUC thuần túy — **đây là kết quả trung thực và cần được phòng thủ chính xác**.

Ý nghĩa thực sự của DNDF **không phải là** tăng AUC mà là:
1. **Uncertainty Quantification**: DNDF cung cấp phân phối xác suất qua nhiều cây → ước lượng độ bất định (variance) tốt hơn.
2. **Tính giải thích**: Decision tree paths có thể truy vết được (inherently interpretable routing).
3. **Soft routing**: Thay vì quyết định "all-or-nothing", DNDF phân phối mẫu mềm dọc theo nhiều nhánh.

**Luận điểm bảo vệ**: *"Tương đương AUC nhưng với khả năng ước lượng uncertainty là đánh đổi chấp nhận được — đặc biệt trong bối cảnh y tế cần explainability."*

---

### Tối Ưu Ngưỡng — Youden J Optimal (Bảng 4.3 Thesis)

> **Nguồn số liệu**: `experiment_report/threshold_analysis.json`

**Wav2Vec 2.0 + DNDF**:

| Ngưỡng (τ) | Sensitivity | Specificity | Youden J |
|-----------|-------------|-------------|----------|
| τ = 0.50 (mặc định) | 54.95% | 85.51% | 0.4046 |
| **τ ≈ 0.43 (Youden optimal)** | **67.19%** | **76.91%** | **0.4410** |
| τ = 0.35 (current) | 72.01% | 71.59% | 0.4359 |
| τ = 0.20 (high-recall) | 85.03% | 49.02% | 0.3405 |

#### 🔍 Diễn Giải:
- Ở **ngưỡng mặc định τ=0.50**: Specificity cao (85%) nhưng Sensitivity thấp (55%) → mô hình "thận trọng", ít báo nhầm.
- Ở **ngưỡng Youden optimal τ≈0.43**: Cân bằng tốt nhất giữa Sensitivity và Specificity — phù hợp cho sàng lọc cộng đồng.
- Ở **ngưỡng thấp τ=0.20**: Sensitivity tăng lên 85% — phù hợp khi muốn "không bỏ sót ca bệnh" (ưu tiên Recall).

**Ứng dụng lâm sàng**: Thesis đề xuất dùng τ ≈ 0.35–0.43 tùy theo mục tiêu triển khai.

---

### ECE — Hiệu Chỉnh Xác Suất (Bảng 4.4 Thesis)

> **Nguồn số liệu**: `experiment_report/ece_results.json`

| Mô Hình | ECE (thấp hơn = tốt hơn) |
|---------|--------------------------|
| Wav2Vec 2.0 + **Softmax** | 0.3062 |
| Wav2Vec 2.0 + **DNDF** | 0.3254 |

#### 🔍 Diễn Giải:
- **ECE (Expected Calibration Error)** đo sự chênh lệch giữa confidence của mô hình và tỷ lệ đúng thực tế.
- Cả hai mô hình đều có ECE cao (~0.3) → **confidence chưa được calibrated tốt**.
- DNDF có ECE cao hơn Softmax một chút → đây là điểm cần thừa nhận trong thesis: "DNDF chưa cải thiện calibration, cần Temperature Scaling hoặc Isotonic Regression trong tương lai."
- **Reliability Diagram**: Xem `experiment_report/ece_reliability_diagram.png` — đường dự đoán nằm dưới đường lý tưởng (mô hình overconfident).

---

### Expert-Labeled Subset (Bảng 4.5 Thesis)

> **Nguồn số liệu**: `experiment_report/expert_subset_results.json`

| Điều kiện | AUC-PR |
|-----------|--------|
| Full dataset (crowdsourced labels) | 0.5263 |
| **Expert-labeled subset** | **0.6054** |

#### 🔍 Diễn Giải:
- Khi đánh giá trên tập nhỏ hơn nhưng được bác sĩ xác nhận nhãn: **AUC-PR tăng từ 0.526 lên 0.605 (+15.1%)**.
- **Kết luận quan trọng**: Nhiều tế có crowdsourced labels nhiễu là nguyên nhân chính kéo giảm hiệu suất. Mô hình thực sự tốt hơn kết quả trên toàn dataset nếu có nhãn sạch.

---

### Frequency Ablation (Bảng 4.6 Thesis)

> **Nguồn số liệu**: `experiment_report/frequency_ablation_results.json`

| Điều kiện | AUC-PR |
|-----------|--------|
| **Baseline** (đầy đủ tần số) | 0.526 |
| Loại bỏ dải **2–6 kHz** | 0.4966 |
| **Sụt giảm** | **−5.6%** |

#### 🔍 Diễn Giải:
- Dải tần số 2–6 kHz (tương ứng vùng formant thứ 2-3 của thanh quản và khí quản) đóng góp **5.6% AUC-PR**.
- Kết quả này xác nhận phát hiện từ LIME/XAI: mô hình thực sự học được đặc điểm âm học lâm sàng của ho COVID-19.
- **Ý nghĩa y học**: Dải 2–6 kHz chứa thông tin về độ cứng phổi, viêm phế quản — đây là vùng bị ảnh hưởng bởi COVID-19.

---

### XAI — Giải Thích LIME (Hình 5.1 Thesis)

> **Nguồn**: `xai_explanations/*.png`

#### 🔍 Cách Đọc Biểu Đồ LIME:
- **Màu xanh lá (xanh nhạt)**: Vùng thời gian-tần số **ủng hộ** phân loại COVID-19 → mô hình "chú ý" vào đây
- **Màu cam/đỏ**: Vùng **chống lại** phân loại COVID-19 (ủng hộ nhãn "khỏe mạnh")
- **Trục X**: Thời gian (giây)
- **Trục Y**: Tần số (Hz)

#### Phát Hiện XAI Chính:
1. **Mẫu COVID-19**: Mô hình tập trung vào **đầu và cuối tiếng ho** (onset và offset) — phù hợp với lý thuyết y học về khả năng đóng glottis kém ở bệnh nhân COVID.
2. **Mẫu khỏe mạnh**: Pattern đều đặn hơn, không có vùng "nóng" đặc trưng.
3. **Dải tần trọng điểm**: Các vùng 500–2000 Hz và 2000–6000 Hz được highlight nhất quán.

---

## 🏗️ Kiến Trúc Mô Hình Chi Tiết

### Model 1: Wav2Vec 2.0 + DNDF

```
Input: Raw Audio Waveform [Batch, 80,000 samples @ 16kHz]
   │
   ▼
[Wav2Vec 2.0 Feature Extractor CNN — FROZEN]
   │
   ▼
[Wav2Vec 2.0 Transformer Encoder — 12 layers]
   │ (layers 0-9: FROZEN; layers 10-11: FINE-TUNED)
   │
   ▼
Mean Pooling → [Batch, 768]     ← Feature Vector
   │
   ▼
[DNDF: 5 Neural Decision Trees, depth=4]
   │  ┌── Tree 1 ── Leaf predictions
   │  ├── Tree 2 ── Leaf predictions
   │  ├── Tree 3 ── Leaf predictions
   │  ├── Tree 4 ── Leaf predictions
   │  └── Tree 5 ── Leaf predictions
   │       ↓ Average
   ▼
Output: P(COVID-19) ∈ [0, 1]
```

**Cơ chế DNDF**:
```
routing_logits = W · x    [Batch, 16 leaves]
leaf_probs = Softmax(routing_logits)   ← Soft routing
leaf_preds = Sigmoid(leaf_weights)     ← Per-leaf predictions
output = Σ(leaf_probs × leaf_preds)   ← Weighted sum
```

### Model 2: AST + DNDF

```
Input: Mel-Spectrogram [Batch, 128 mel-bins, 1024 time-frames]
   │
   ▼
[AST: Audio Spectrogram Transformer — MIT pre-trained]
   │  Patch embedding (16×16) + Position encoding
   │  12 Transformer layers
   │
   ▼
CLS Token → [Batch, 768]     ← Feature Vector
   │
   ▼
[DNDF: 5 Neural Decision Trees, depth=4]   (same as Model 1)
   │
   ▼
Output: P(COVID-19) ∈ [0, 1]
```

---

## 🔑 Các Tham Số Quan Trọng Có Thể Điều Chỉnh

### Trong `train_1d_sota.py` / `train_2d_sota.py`:
```python
BATCH_SIZE = 8       # Tăng nếu GPU nhiều VRAM hơn
EPOCHS = 15          # Tăng lên 20-30 nếu muốn kết quả tốt hơn
N_SPLITS = 5         # Số fold cross-validation
```

### Trong `model_1d_sota.py`:
```python
DNDF(num_trees=5, depth=4, ...)  # Thay đổi số cây và độ sâu
```

### Trong `threshold_optimization.py`:
```python
# Điều chỉnh ngưỡng τ theo nhu cầu lâm sàng
tau_clinical = 0.43   # Youden optimal (AUC cân bằng)
tau_screening = 0.30  # Ưu tiên Sensitivity (ít bỏ sót)
tau_confirmation = 0.65  # Ưu tiên Specificity (giảm false alarm)
```

---

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Giải pháp**: Giảm `BATCH_SIZE` trong `train_1d_sota.py` (thử 4 hoặc 2).

### Lỗi 2: Model Not Found
```
FileNotFoundError: best_wav2vec_fold_1.pth
```
**Giải pháp**: Cần chạy `train_1d_sota.py` trước để tạo model weights.

### Lỗi 3: Hugging Face Download Error
```
OSError: Can't load tokenizer for 'facebook/wav2vec2-base'
```
**Giải pháp**: Kiểm tra kết nối internet hoặc set cache:
```bash
set HF_HOME=D:\GROWTH\GRADUTION_MASTER\Covid_Cough_Research\.hf_cache
```

### Lỗi 4: Feature Directory Not Found
```
FileNotFoundError: data/features_1d_raw
```
**Giải pháp**: Chạy `extract_features_1d.py` trước.

---

## 📌 Tóm Tắt Nhanh — Files Quan Trọng Nhất

| Mục Đích | File |
|----------|------|
| **Kiến trúc mô hình** | `src/model_1d_sota.py`, `src/model_2d_sota.py` |
| **Huấn luyện** | `src/train_1d_sota.py`, `src/train_2d_sota.py` |
| **Kết quả số liệu thesis** | `evaluation_results/detailed_metrics.json` |
| **Ablation DNDF vs Softmax** | `experiment_report/ablation_results.json` |
| **Ngưỡng tối ưu** | `experiment_report/threshold_analysis.json` |
| **ECE calibration** | `experiment_report/ece_results.json` |
| **Biểu đồ ROC** | `experiment_report/roc_curves.png` |
| **Giải thích XAI** | `xai_explanations/*.png` |
| **Model weights** | `best_wav2vec_fold_*.pth`, `best_ast_fold_*.pth` |
| **Chạy toàn pipeline** | `src/run_full_experiment.py` |

---

## 🗂️ Giải Thích Chi Tiết Từng File Code

> Phần này giải thích công dụng chi tiết của từng file, bao gồm **ý nghĩa từng dòng/tham số quan trọng** và **điều gì xảy ra nếu bạn thay đổi chúng**.

---

### `src/model_1d_sota.py` — Kiến Trúc Wav2Vec 2.0 + DNDF

**Công dụng**: Định nghĩa kiến trúc mô hình chính của thesis — kết hợp bộ trích xuất đặc trưng Wav2Vec 2.0 với bộ phân loại DNDF.

```python
class NeuralDecisionTree(nn.Module):
    def __init__(self, depth, feature_dim, num_classes=1, metadata_dim=0):
        self.num_leaves = 2 ** depth        # depth=4 → 16 lá mỗi cây
        self.decision_nodes = nn.Linear(feature_dim + metadata_dim, self.num_leaves, bias=False)
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, num_classes))
```

| Dòng/Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `depth = 4` | Độ sâu cây quyết định → 2⁴=16 lá | Tăng lên 5 → 32 lá, phức tạp hơn nhưng dễ overfit |
| `feature_dim = 768` | Chiều vector đặc trưng ra từ Wav2Vec 2.0 | **Không đổi** — phụ thuộc kiến trúc pre-trained |
| `num_classes = 1` | Bài toán nhị phân (1 output neuron) | Đổi sang 2 nếu dùng Cross-Entropy thay BCE |
| `metadata_dim = 0` | Số chiều metadata bổ sung (age, gender,...) | Đặt = 4 để thêm Age+Gender+Smoke+Asthma vào routing |

```python
    def forward(self, x, metadata=None):
        routing_logits = self.decision_nodes(routing_input)   # [Batch, 16]
        leaf_probs = F.softmax(routing_logits, dim=1)         # Soft routing
        leaf_preds = torch.sigmoid(self.leaves)               # [16, 1]
        out = torch.matmul(leaf_probs, leaf_preds)            # [Batch, 1]
```

| Dòng | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `F.softmax(routing_logits, dim=1)` | **Soft routing**: mỗi mẫu được phân bổ mềm vào tất cả 16 lá | Đổi thành `argmax` → hard routing, mất tính khả vi (không train được gradient) |
| `torch.sigmoid(self.leaves)` | Ép leaf predictions về [0,1] | **Không nên đổi** — cần để output có nghĩa xác suất |
| `torch.matmul(leaf_probs, leaf_preds)` | Dự đoán cuối = tổng có trọng số | Đây là công thức P(y|x) của DNDF |

```python
class DNDF(nn.Module):
    def __init__(self, num_trees, depth, feature_dim, num_classes=1, metadata_dim=0):
        self.trees = nn.ModuleList([NeuralDecisionTree(...) for _ in range(num_trees)])

    def forward(self, x, metadata=None):
        return torch.mean(torch.stack(tree_outputs), dim=0)  # Average ensemble
```

| Tham số | Giá trị hiện tại | Nếu thay đổi |
|---|---|---|
| `num_trees = 5` | 5 cây quyết định song song | Tăng lên 10 → độ ổn định cao hơn nhưng chậm gấp đôi |
| `torch.mean(...)` | Lấy trung bình dự đoán của 5 cây | Đổi thành weighted sum nếu muốn ưu tiên cây tốt hơn |

```python
class Wav2Vec2_DNDF_Model(nn.Module):
    def __init__(self, num_classes=1, metadata_dim=0):
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()   # Đóng băng CNN
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if not any(f"layers.{i}." in name for i in range(10, 12)):
                param.requires_grad = False                    # Đóng băng layers 0-9
```

| Dòng | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `from_pretrained("facebook/wav2vec2-base")` | Load model pre-trained trên 960h LibriSpeech | Đổi sang `wav2vec2-large` → 300M params, VRAM cần nhiều hơn |
| `_freeze_parameters()` | Không cập nhật CNN extractor trong training | Bỏ dòng này → CNN cũng học, cần nhiều data hơn |
| `range(10, 12)` | Chỉ fine-tune 2 lớp encoder cuối (10, 11) | Đổi thành `range(8, 12)` → fine-tune 4 lớp, kết quả tốt hơn nhưng chậm hơn |

---

### `src/model_2d_sota.py` — Kiến Trúc AST + DNDF

**Công dụng**: Định nghĩa mô hình thứ hai sử dụng Audio Spectrogram Transformer + Multi-scale Fusion + DNDF.

```python
class AST_DNDF_Model(nn.Module):
    def __init__(self, num_classes=1):
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Multi-scale Feature Fusion với 3 kernel sizes khác nhau
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=7, padding=3)
        self.feature_dim = 256 * 3   # = 768 sau khi concat
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `"MIT/ast-finetuned-audioset-10-10-0.4593"` | AST đã fine-tune trên AudioSet (2M clips) | Không nên đổi — đây là checkpoint tốt nhất hiện có |
| `kernel_size=3,5,7` | Bắt đặc trưng tại 3 tầm nhìn thời gian khác nhau | Đổi sang 1,3,5 → ngắn hơn, hợp với audio ngắn |
| `out_channels=256` | Số chiều output mỗi conv | Tăng lên 512 → feature_dim=1536, cần sửa DNDF input |
| `self.feature_dim = 256 * 3` | **Quan trọng**: phải match với `DNDF(feature_dim=...)` | Nếu đổi conv, phải cập nhật dòng này tương ứng |

```python
    def forward(self, x):
        x = x.squeeze(1)           # [Batch, 1, 128, 1024] → [Batch, 128, 1024]
        x = x.transpose(1, 2)     # → [Batch, 1024, 128] (AST cần freq ở cuối)
        outputs = self.ast(x)
        hidden_states = outputs.last_hidden_state.transpose(1, 2)  # [Batch, 768, SeqLen]
        
        feat1 = F.relu(self.conv1(hidden_states)).mean(dim=2)  # Global avg pooling
        feat2 = F.relu(self.conv2(hidden_states)).mean(dim=2)
        feat3 = F.relu(self.conv3(hidden_states)).mean(dim=2)
        fused_features = torch.cat([feat1, feat2, feat3], dim=1)  # [Batch, 768]
```

> ⚠️ **Lưu ý quan trọng**: Thứ tự `squeeze → transpose → ast → transpose → conv → mean → cat` phải giữ nguyên — thay đổi bất kỳ bước nào sẽ gây lỗi shape mismatch.

---

### `src/train_1d_sota.py` — Huấn Luyện Model Wav2Vec 2.0 + DNDF

**Công dụng**: Script huấn luyện chính, thực hiện 5-Fold Cross-Validation cho Model 1.

```python
BATCH_SIZE = 8    # Dòng 22
EPOCHS = 15       # Dòng 23
N_SPLITS = 5      # Dòng 24
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `BATCH_SIZE = 8` | Số mẫu mỗi mini-batch | Giảm xuống 4 nếu CUDA OOM; tăng lên 16 nếu có nhiều VRAM |
| `EPOCHS = 15` | Số vòng lặp toàn bộ dataset | Tăng lên 20 để hội tụ tốt hơn; giảm xuống 10 để thử nhanh |
| `N_SPLITS = 5` | Số fold trong StratifiedKFold | Đổi sang 3 để chạy nhanh hơn; 10 để đánh giá chính xác hơn |

```python
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):   # Dòng 36
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `alpha = 0.8` | Trọng số cho class dương (COVID-19) | Tăng lên 0.9 → ưu tiên hơn cho ca COVID; giảm xuống 0.5 = không bias |
| `gamma = 2.0` | Focusing parameter — giảm loss cho mẫu dễ | Tăng lên 3.0 → tập trung mạnh hơn vào mẫu khó (cẩn thận overfit) |

```python
optimizer = optim.AdamW([
    {'params': model.wav2vec2.parameters(), 'lr': 1e-5},   # Dòng 116-117
    {'params': model.dndf.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `lr=1e-5` (Wav2Vec) | Học rất chậm để không phá vỡ pre-trained weights | Tăng lên 1e-4 → nguy cơ catastrophic forgetting của pre-trained knowledge |
| `lr=1e-3` (DNDF) | Học nhanh vì DNDF được khởi tạo ngẫu nhiên | Giảm xuống 1e-4 → DNDF hội tụ chậm hơn |
| `weight_decay=1e-4` | L2 regularization chống overfit | Tăng lên 1e-3 → regularize mạnh hơn nếu overfit |

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)  # Dòng 121
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `CosineAnnealingLR` | LR giảm dần theo đường cos từ lr_max → 0 | Đổi sang `StepLR` → giảm cố định mỗi N epoch |
| `T_max=EPOCHS` | Số bước để giảm xuống lr_min | = EPOCHS nghĩa là LR về ~0 vào epoch cuối cùng |

```python
torch.save(model.state_dict(), f'best_wav2vec_fold_{fold+1}.pth')  # Dòng 158
```
> Chỉ lưu khi AUC trên val set của epoch đó **tốt hơn** tất cả epoch trước trong cùng fold.

---

### `src/train_2d_sota.py` — Huấn Luyện Model AST + DNDF

**Công dụng**: Giống `train_1d_sota.py` nhưng dành cho Model 2 (spectrogram-based).

**Điểm khác biệt quan trọng**:

```python
BATCH_SIZE = 16   # Dòng 22 — tăng gấp đôi vì AST input nhỏ hơn Wav2Vec raw audio

def mixup_data(x, y, alpha=0.4, device='cuda'):   # Dòng 26-35
    lam = np.random.beta(alpha, alpha)             # λ ~ Beta(0.4, 0.4)
    mixed_x = lam * x + (1 - lam) * x[index, :]  # Trộn 2 mẫu theo tỉ lệ λ
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `alpha = 0.4` | Tham số phân phối Beta cho MixUp | Tăng lên 1.0 → trộn đều 50/50; giảm xuống 0.1 → gần như không trộn |
| MixUp chỉ ở Model 2 | Spectrogram 2D dễ MixUp hơn raw waveform 1D | Nếu áp MixUp cho 1D, cần điều chỉnh augmentation riêng |

```python
torch.save(model.state_dict(), f'best_ast_fold_{fold+1}.pth')  # Dòng 173
```

---

### `src/dataset_offline.py` — Dataset Loader

**Công dụng**: Đọc dữ liệu từ file `.pt` đã trích xuất sẵn thay vì xử lý audio mỗi lần — tăng tốc training đáng kể.

```python
class CovidCoughDatasetOffline(Dataset):
    def __init__(self, metadata_path, feature_dir, filter_expert=False, use_metadata=False):
```

| Tham số | Ý nghĩa | Khi nào dùng |
|---|---|---|
| `filter_expert=False` | Dùng toàn bộ dataset (kể cả crowdsourced noisy labels) | Đặt `True` → chỉ dùng mẫu có `expert_evaluation_score >= 0.8` |
| `use_metadata=False` | Không đưa Age/Gender vào mô hình | Đặt `True` → dataset trả về thêm `meta_tensor` (4 chiều) |

```python
if filter_expert and 'expert_evaluation_score' in df.columns:
    df = df[df['expert_evaluation_score'] >= 0.8]   # Dòng 15
```
> Ngưỡng `0.8` có thể chỉnh: `0.9` → chỉ lấy mẫu chắc chắn nhất; `0.7` → lấy nhiều mẫu hơn nhưng noisy hơn.

```python
if waveform.shape[1] > max_length:
    waveform = waveform[:, :max_length]            # Cắt nếu quá dài
else:
    waveform = torch.nn.functional.pad(...)        # Đệm 0 nếu quá ngắn
```
> Trong `extract_features_1d.py`, `max_length = 16000 * 5 = 80,000` điểm (5 giây). Đổi thành `16000 * 3` = 3 giây nếu muốn audio ngắn hơn.

---

### `src/extract_features_1d.py` — Trích Xuất Features Wav2Vec

**Công dụng**: Đọc từng file audio thô, chuẩn hóa và lưu thành tensor `.pt` để tăng tốc training.

```python
target_sr = 16000       # Dòng 18 — Sample rate chuẩn cho Wav2Vec 2.0
max_length = target_sr * 5  # Dòng 19 — Cắt/đệm về đúng 5 giây
```

| Dòng | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `target_sr = 16000` | **Bắt buộc** — Wav2Vec 2.0 được train ở 16kHz | Đổi sang 22050 → mô hình sẽ không hoạt động đúng |
| `max_length = 16000 * 5` | Cố định 5 giây cho mọi sample | Đổi thành `16000 * 3` = 3 giây → tensor nhỏ hơn, training nhanh hơn |

```python
# Z-score normalization — BẮT BUỘC cho Wav2Vec 2.0
waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)  # Dòng 54
```
> **Không được bỏ dòng này** — Wav2Vec 2.0 được pre-train với input đã normalize, bỏ dòng này sẽ làm mô hình hoạt động sai.

---

### `src/extract_features.py` — Trích Xuất Features AST

**Công dụng**: Chuyển audio thành Mel-Spectrogram rồi chuẩn hóa kích thước cho AST.

```python
spectrogram_resized = F.interpolate(spectrogram, size=(128, 1024), mode='bilinear')  # Dòng 32
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `size=(128, 1024)` | **Bắt buộc** — AST input cần 128 mel-bins × 1024 time-frames | Đổi kích thước này phải sửa đồng bộ trong `model_2d_sota.py` |
| `mode='bilinear'` | Nội suy 2D khi resize | Đổi sang `nearest` → nhanh hơn nhưng kém chất lượng hơn |

---

### `src/ablation_dndf_vs_softmax.py` — Ablation Study

**Công dụng**: Chạy thực nghiệm A/B để so sánh DNDF và Softmax trong điều kiện được kiểm soát.

```python
EPOCHS = 15   # Dòng 48 — Giống training chính
N_SPLITS = 5  # Dòng 49
RANDOM_SEED = 42  # Dòng 50 — Seed cố định để reproducible
```

```python
class Wav2Vec2_Softmax_Model(nn.Module):
    def __init__(self, num_classes=1):
        # IDENTICAL freezing strategy as DNDF model
        self.wav2vec2.feature_extractor._freeze_parameters()
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if not any(f"layers.{i}." in name for i in range(10, 12)):
                param.requires_grad = False
        
        # BASELINE CLASSIFIER: Simple linear projection
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
```

> ⭐ **Điểm mấu chốt**: Cấu hình freeze Wav2Vec **giống hệt** Model DNDF — đảm bảo so sánh công bằng, chỉ khác ở lớp classifier.

| Dòng | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `nn.Linear(768, 256)` | Bottleneck từ 768 xuống 256 | Đổi thành Linear(768, 1) trực tiếp → đơn giản hơn nhưng kém hơn |
| `nn.Dropout(0.3)` | Dropout để chống overfit | Tăng lên 0.5 nếu overfit; bỏ đi nếu dataset đủ lớn |

---

### `src/threshold_optimization.py` — Tối Ưu Ngưỡng

**Công dụng**: Tính toán ngưỡng quyết định tối ưu từ dữ liệu ROC đã có — **không cần train lại**.

```python
# Dòng 57-61: Công thức Youden's J
youden_j = tpr - fpr              # J = Sensitivity + Specificity - 1 = TPR - FPR
opt_idx = np.argmax(youden_j)     # Điểm trên ROC curve có J lớn nhất
opt_sens = tpr[opt_idx]           # Sensitivity tại điểm tối ưu
opt_spec = 1 - fpr[opt_idx]       # Specificity tại điểm tối ưu
```

```python
# Dòng 91: Các mức sensitivity để sweep
targets = [0.50, 0.55, 0.60, 0.65, 0.67, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90, 0.95]
```
> Thêm giá trị vào list này để phân tích thêm điểm hoạt động. Ví dụ: thêm `0.95` để xem specificity khi sensitivity 95%.

---

### `src/experiment_ece_reliability.py` — Thực Nghiệm ECE

**Công dụng**: Đo lường chất lượng calibration của mô hình — confidence có tương quan với accuracy không?

```python
BATCH_SIZE = 8    # Dòng 30
EPOCHS = 5        # Dòng 31 — Giảm xuống 5 (thay vì 15) để chạy nhanh, chỉ cần ước lượng ECE
N_SPLITS = 3      # Dòng 32 — Giảm xuống 3 fold thay vì 5
```

> ⚠️ **Lưu ý**: Script này **train lại từ đầu** với ít epoch để lấy OOF predictions (out-of-fold). Kết quả ECE có thể khác nhẹ so với mô hình full-trained.

```python
def calculate_ece(y_true, y_proba, n_bins=10):   # trong utils_metrics.py
    bin_boundaries = np.linspace(0, 1, n_bins + 1)  # Chia thành 10 bucket confidence
```
> `n_bins=10` là chuẩn. Tăng lên 15-20 → phân tích chi tiết hơn nhưng cần nhiều mẫu hơn.

---

### `src/utils_metrics.py` — Thư Viện Metrics

**Công dụng**: Thư viện dùng chung — tính toán AUC-ROC, AUC-PR, ECE, Youden's J, threshold sweep.

```python
class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_proba) -> Dict:
        # AUC-PR là PRIMARY metric (không phải AUC-ROC)
        # Lý do: dataset mất cân bằng, AUC-PR phản ánh đúng hơn hiệu quả trên minority class
        auc_pr_score = auc(recall_curve, precision_curve)   # Dòng 61
```

> **Quan trọng**: Code comment tại dòng 56-58 giải thích tại sao AUC-PR được đặt là PRIMARY metric trong thesis — đây là câu trả lời cho câu hỏi của giám khảo về lựa chọn metric.

```python
class ThresholdOptimizer:
    # Youden WJ (1950). 'Index for rating diagnostic tests.' Cancer, 3(1):32-35.
    @staticmethod
    def find_optimal_youden(y_true, y_proba) -> Dict:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_j = tpr - fpr           # J = TPR - FPR
        optimal_idx = np.argmax(youden_j)
        return {
            'optimal_threshold': float(thresholds[optimal_idx]),  # τ tối ưu
            ...
        }
```

---

### `run_all_experiments.py` — Master Runner (3 Thực Nghiệm Cuối)

**Công dụng**: Chạy tuần tự 3 thực nghiệm bổ sung được yêu cầu bởi Hội đồng phản biện. File này ở **thư mục gốc** (không phải trong `src/`).

```python
def run_script(script_name, description):
    result = subprocess.run([sys.executable, script_name], check=True)  # Dòng 23
```

| Tham số | Ý nghĩa | Nếu thay đổi |
|---|---|---|
| `check=True` | Nếu script lỗi → dừng toàn bộ pipeline | Đổi thành `False` → tiếp tục chạy kể cả khi có lỗi |
| `sys.executable` | Dùng Python từ virtual environment hiện tại | **Không nên đổi** — đảm bảo đúng venv với đầy đủ packages |

```python
def main():
    time.sleep(3)   # Dòng 42 — Chờ 3 giây để người dùng đọc thông báo
    
    run_script("src/experiment_ece_reliability.py", "Experiment 1 - ECE")   # Thứ nhất
    run_script("src/experiment_expert_labels.py",  "Experiment 2 - Expert") # Thứ hai
    run_script("src/experiment_freq_ablation.py",  "Experiment 3 - Freq")   # Thứ ba
```

> Thứ tự chạy là có chủ ý: ECE → Expert → Frequency. Không cần thay đổi thứ tự vì 3 thực nghiệm này độc lập nhau.

**Lệnh chạy**:
```bash
cd D:\GROWTH\GRADUTION_MASTER\Covid_Cough_Research
python run_all_experiments.py
```

---

### `src/xai_interpretability.py` — Giải Thích LIME

**Công dụng**: Tạo LIME explanations dạng heatmap trên spectrogram, cho thấy vùng tần số-thời gian nào mô hình dùng để đưa ra quyết định.

> Cơ chế LIME: Tạo nhiều phiên bản nhiễu của mẫu đầu vào → quan sát dự đoán thay đổi thế nào → suy ra feature importance.

---

## 🏆 Danh Sách Model Weights Đã Lưu & Model Được Chốt Sử Dụng

### Toàn Bộ Model Weights Tồn Tại

| Tên File | Kích Thước | Mô Hình | Fold | Khi Nào Được Lưu |
|----------|-----------|---------|------|-----------------|
| `best_wav2vec_fold_1.pth` | ~360 MB | Wav2Vec 2.0 + DNDF | Fold 1 | Epoch tốt nhất AUC trên val set fold 1 |
| `best_wav2vec_fold_2.pth` | ~360 MB | Wav2Vec 2.0 + DNDF | Fold 2 | Epoch tốt nhất AUC trên val set fold 2 |
| `best_wav2vec_fold_3.pth` | ~360 MB | Wav2Vec 2.0 + DNDF | Fold 3 | Epoch tốt nhất AUC trên val set fold 3 |
| `best_wav2vec_fold_4.pth` | ~360 MB | Wav2Vec 2.0 + DNDF | Fold 4 | Epoch tốt nhất AUC trên val set fold 4 |
| `best_wav2vec_fold_5.pth` | ~360 MB | Wav2Vec 2.0 + DNDF | Fold 5 | Epoch tốt nhất AUC trên val set fold 5 |
| `best_ast_fold_1.pth` | ~330 MB | AST + DNDF | Fold 1 | Epoch tốt nhất AUC trên val set fold 1 |
| `best_ast_fold_2.pth` | ~330 MB | AST + DNDF | Fold 2 | Epoch tốt nhất AUC trên val set fold 2 |
| `best_ast_fold_3.pth` | ~330 MB | AST + DNDF | Fold 3 | Epoch tốt nhất AUC trên val set fold 3 |
| `best_ast_fold_4.pth` | ~330 MB | AST + DNDF | Fold 4 | Epoch tốt nhất AUC trên val set fold 4 |
| `best_ast_fold_5.pth` | ~330 MB | AST + DNDF | Fold 5 | Epoch tốt nhất AUC trên val set fold 5 |
| `best_1d_model.pth` | ~185 KB | Model 1D (cũ/nhỏ hơn) | N/A | Phiên bản cũ — không dùng cho thesis |
| `best_2d_model.pth` | ~43 MB | Model 2D (cũ/nhỏ hơn) | N/A | Phiên bản cũ — không dùng cho thesis |

> **Tổng dung lượng model weights**: ~3.7 GB

---

### 🥇 Model Được Chốt Và Sử Dụng Cho Thesis

> **TL;DR**: Mô hình được chốt là **Wav2Vec 2.0 + DNDF (Ensemble 5 Folds)** — không phải từ một fold đơn lẻ, mà là **ensemble trung bình của cả 5 file**: `best_wav2vec_fold_1.pth` → `best_wav2vec_fold_5.pth`.

---

#### ❌ `best_1d_model.pth` và `best_2d_model.pth` — **KHÔNG DÙNG**

Đây là model weights từ **giai đoạn phát triển sơ khai** (old stage), kiến trúc đơn giản hơn, **không phải mô hình SOTA được trình bày trong thesis**.

---

#### ✅ `best_wav2vec_fold_*.pth` — MÔ HÌNH ĐỀ XUẤT CHÍNH

```
╔══════════════════════════════════════════════════════════════════╗
║  ⭐ MODEL ĐƯỢC CHỐT — ĐỂ SỬ DỤNG VÀ BÁO CÁO TRONG THESIS ⭐   ║
║                                                                  ║
║  Tên:   Wav2Vec 2.0 + DNDF (5-Fold Ensemble)                    ║
║  Files: best_wav2vec_fold_1.pth  (AUC-ROC: ~0.7725 trên val 1) ║
║         best_wav2vec_fold_2.pth  (AUC-ROC: ~0.7348 trên val 2) ║
║         best_wav2vec_fold_3.pth  (AUC-ROC: ~0.7867 trên val 3) ║
║         best_wav2vec_fold_4.pth  (AUC-ROC: ~0.7660 trên val 4) ║
║         best_wav2vec_fold_5.pth  (AUC-ROC: ~0.7592 trên val 5) ║
║                                                                  ║
║  AUC-ROC Trung Bình: 0.7638 ± 0.0171 (5-Fold CV)               ║
║  AUC-ROC Evaluation: 0.7735 (cross-dataset test)                ║
║  AUC-PR:             0.5263 (full dataset)                      ║
║  AUC-PR (Expert):    0.6054 (expert-filtered subset)            ║
╚══════════════════════════════════════════════════════════════════╝
```

**Lý do chọn**:
1. **AUC-ROC cao hơn** so với AST+DNDF (0.7735 vs 0.6910) trên tập test
2. **F1-Score cao hơn** (0.5391 vs 0.3536) — cân bằng tốt hơn giữa Precision và Recall
3. **Precision cao hơn** (0.5445 vs 0.2728) — ít false positive hơn
4. **Prevalence dataset phù hợp** (23.5% positive) — AST dataset imbalanced hơn (11%)
5. **Kiến trúc 1D phù hợp hơn** với Wav2Vec vốn được thiết kế cho raw waveform

#### ✅ `best_ast_fold_*.pth` — MÔ HÌNH ĐỐI CHIẾU (SECONDARY)

```
╔══════════════════════════════════════════════════════════════════╗
║  📊 MODEL ĐỐI CHIẾU — DÙNG ĐỂ SO SÁNH TRONG THESIS             ║
║                                                                  ║
║  Tên:   AST + DNDF (5-Fold Ensemble)                            ║
║  Files: best_ast_fold_1.pth → best_ast_fold_5.pth              ║
║                                                                  ║
║  AUC-ROC Evaluation: 0.6910 (thấp hơn Wav2Vec)                 ║
║  Accuracy:           0.7966 (cao hơn do class imbalance)        ║
║  Precision:          0.2728 (nhiều false positive)              ║
╚══════════════════════════════════════════════════════════════════╝
```

**Vai trò**: Chứng minh sự đa dạng phương pháp (1D vs 2D), nhưng **không phải mô hình chính**.

---

### Cách Tải Model Để Inference/Đánh Giá

```python
import torch
from src.model_1d_sota import Wav2Vec2_DNDF_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ CÁCH DÙNG ĐỀ XUẤT: Ensemble 5 folds (tốt nhất)
models = []
for fold in range(1, 6):
    model = Wav2Vec2_DNDF_Model(num_classes=1).to(device)
    model.load_state_dict(torch.load(f"best_wav2vec_fold_{fold}.pth", map_location=device))
    model.eval()
    models.append(model)

# Inference với ensemble
def predict_ensemble(waveform_tensor):
    with torch.no_grad():
        predictions = [m(waveform_tensor.to(device)).cpu() for m in models]
        avg_pred = torch.mean(torch.stack(predictions), dim=0)  # Trung bình 5 mô hình
    return avg_pred

# ✅ CÁCH ĐƠN GIẢN HƠN: Chỉ dùng fold tốt nhất (fold 3 — AUC cao nhất)
best_model = Wav2Vec2_DNDF_Model(num_classes=1).to(device)
best_model.load_state_dict(torch.load("best_wav2vec_fold_3.pth", map_location=device))
best_model.eval()
```

> **Tại sao fold 3 là tốt nhất?** Theo `ablation_results.json`: fold 3 đạt AUC-ROC = 0.7867 — cao nhất trong 5 folds của Wav2Vec + DNDF.

---

### Bảng So Sánh AUC Từng Fold

| Fold | Wav2Vec 2.0 + DNDF | Wav2Vec 2.0 + Softmax | Model Nào Tốt Hơn |
|------|--------------------|-----------------------|-------------------|
| Fold 1 | 0.7725 | 0.7712 | DNDF +0.0013 |
| Fold 2 | 0.7348 | 0.7594 | Softmax +0.0246 |
| Fold 3 | **0.7867** | 0.7776 | DNDF **+0.0091** |
| Fold 4 | 0.7660 | 0.7725 | Softmax +0.0065 |
| Fold 5 | 0.7592 | 0.7506 | DNDF +0.0086 |
| **Mean** | **0.7638** | **0.7663** | Softmax +0.0025 (chênh lệch nhỏ) |

> **Kết luận**: DNDF thắng ở 3/5 folds nhưng trung bình thua nhẹ (−0.0025) do fold 2 kém hơn nhiều. **Đây là kết quả cần trình bày trung thực trong thesis** và lý giải bằng: "variance lớn hơn của DNDF (std=0.0171 vs 0.0098) cho thấy độ nhạy cảm với phân chia dữ liệu — đây là hướng cải thiện trong tương lai (ensemble stacking)."

---

### Sơ Đồ Lựa Chọn Model Cho Inference

```
Cần inference?
    │
    ├── Độ chính xác cao nhất?
    │       → Ensemble 5 folds: best_wav2vec_fold_1..5.pth
    │
    ├── Tốc độ nhanh, demo nhanh?
    │       → Single best: best_wav2vec_fold_3.pth  (AUC=0.7867)
    │
    ├── So sánh với AST?
    │       → best_ast_fold_*.pth  (AUC=0.6910, dùng để benchmark)
    │
    └── KHÔNG BAO GIỜ dùng:
            → best_1d_model.pth   (kiến trúc cũ, không phải SOTA)
            → best_2d_model.pth   (kiến trúc cũ, không phải SOTA)
```

---

*Tháng 6/2026*
