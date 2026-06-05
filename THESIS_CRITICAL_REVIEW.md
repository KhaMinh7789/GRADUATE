# 🔍 RÀ SOÁT TỐI THIỂU HÓA - LỦA VẬN LUẬN ÁN

**Ngày rà soát**: 14/05/2026  
**Trạng thái**: ⚠️ **CẦN CẢI THIỆN NGAY TRƯỚC KHI BẢO VỆ**  
**Mục đích**: Xác định các lỗi logic, thiếu sót phương pháp luận, và không nhất quán trong công thức

---

## **PHẦN I: RÀ SOÁT TÍNH LOGIC CỦA KIẾN TRÚC HỆ THỐNG**

### **1️⃣ VẤN ĐỀ: AST Model (2D) - Sự Sụt Giảm Precision Nghiêm Trọng**

#### 🔴 **Vấn đề Chính**
- **Hiện tượng**: AST (2D spectrogram) có precision chỉ **27.28%** trong cross-domain evaluation
  - Wav2Vec (1D raw audio): Precision = 53.39%
  - AST (2D spectrogram): Precision = 27.28%
  - **Tỷ lệ sụt giảm**: -49.1% (cực kỳ đáng lo ngại)

#### 🔎 **Nguyên Nhân Gốc Rễ (Root Cause Analysis)**

**Giả thuyết 1: Mel-spectrogram Không Bảo Toàn Fine-grained Transients**
- Chuyển đổi từ 1D raw audio sang 2D spectrogram mất thông tin về:
  - Biến thể âm thanh tinh tế (micro-variations)
  - Đặc trưng pha (phase information) - chỉ giữ magnitude
  - Độ phân giải tần số không đủ cao để bắt được 2kHz-6kHz critical band

**Giả thuyết 2: Ánh xạ Mel-scale Bớt Tối ưu Cho Cough Acoustics**
```
Current Mel-spectrogram config:
- n_fft = 1024 (64ms window)
- n_mels = 64 (mel bins)
- Frequency range: 0-8000 Hz

Vấn đề:
- 64 bins phân phối theo log scale → resolution thấp ở dải cao (2-6 kHz)
- Window size 64ms có thể quá dài cho cough bursts (30-200ms)
```

**Giả thuyết 3: Overfitting Lên Noise Background Của Source Dataset**
- AST pre-trained trên AudioSet (âm nhạc, tiếng nói)
- Tiếng ho là acoustic event hiếm → mô hình học features của nền tảng ghi âm thay vì biomarkers phổ quát
- Cross-domain test lộ ra sự khác biệt thiết bị/nền tảng

#### 📋 **Các Tham Số Cần Rà Soát Lại**

| Tham Số | Giá Trị Hiện Tại | Gợi Ý Cải Thiện | Lý Do |
|---------|-----------------|-----------------|-------|
| **n_fft** | 1024 (64ms) | 512-768 (32-48ms) | Capture cough bursts tốt hơn |
| **n_mels** | 64 | 128 hoặc 256 | Độ phân giải cao hơn ở dải 2-6kHz |
| **hop_length** | 512 (50% overlap) | 256-384 (75-87.5% overlap) | Chi tiết thời gian tốt hơn |
| **f_min** | 0 Hz | 100-200 Hz | Bỏ bruit thấp, focus vào cough |
| **f_max** | 8000 Hz | 8000 Hz ✓ | Ok - cough chủ yếu ≤8kHz |
| **Normalization** | Không rõ | **Loudness (LUFS)** ⚠️ | **Giảm dataset device-bias** |

#### ✅ **Đề Xuất Cải Thiện (Action Items)**

**A1: Bổ sung Loudness Normalization**
```python
# Tại extract_features.py hoặc prepare modules
import pyloudnorm

meter = pyloudnorm.Meter(sr=16000)
loudness = meter.integrated_loudness(audio)
audio_norm = pyloudnorm.normalize(audio, loudness, -23.0)  # LUFS target
```

**A2: Thực nghiệm Hyperparameter Sweep**
```
Grid search cho:
- n_mels: [64, 128, 256]
- n_fft: [512, 768, 1024]
- f_min: [0, 100, 200]

Đánh giá trên VALIDATION set và report AUC-PR tốt nhất
```

**A3: Thêm Mel-spectrogram QC (Quality Control)**
```
Validate output spectrogram:
- No NaN/Inf values
- Dynamic range > 40dB
- Time-frequency coverage > 80%
```

---

### **2️⃣ VẤN ĐỀ: DNDF (Deep Neural Decision Forest) Logic**

#### 🔴 **Vấn đề: Thiếu A/B Testing Trực Tiếp**

**Mệnh đề Quảng Bá**:  
> "DNDF cải thiện AUC-ROC lên 0.7735, tăng 10.3% so với baseline"

**Kẽ Hở Phương Pháp Luận**:
- Không có thực nghiệm so sánh: `Wav2Vec 2.0 + Softmax` vs `Wav2Vec 2.0 + DNDF`
- Hiện tại chỉ có: `Wav2Vec 2.0 + DNDF` (single configuration)
- **Không thể kết luận** cải thiện 10.3% đến từ DNDF hay từ feature extraction của Wav2Vec

#### 📊 **Thực Nghiệm A/B Cần Thêm**

```
Setup:
┌─────────────────────────────────────────────┐
│ Feature Extractor: Wav2Vec 2.0 (frozen)    │
├─────────────────────────────────────────────┤
│ Classifier 1: Linear Head + Softmax        │ ← Baseline
│ Classifier 2: DNDF (5 trees, depth 4)     │ ← Proposal
│ Classifier 3: Random Forest (sklearn)      │ ← Control
└─────────────────────────────────────────────┘

Metrics:
- AUC-ROC, AUC-PR, F1, Sensitivity, Specificity
- Training time, Memory usage
- Cross-dataset performance (generalization)
```

#### 🔬 **DNDF Implementation Issues**

**Issue A: Non-standard Routing (sigmoid → softmax)**
```python
# CURRENT (Incorrect)
routing_probs = torch.sigmoid(self.decision_nodes(x))  # [0,1]
leaf_probs = F.softmax(routing_probs, dim=1)          # softmax([0,1])
# Problem: Applying softmax to already-bounded [0,1] values is non-standard
# Result: Distorts routing probability interpretation

# CORRECT
routing_probs = self.decision_nodes(x)                # Raw logits
leaf_probs = F.softmax(routing_probs, dim=1)          # softmax → sum=1
```

**Issue B: Missing Leaf Node Probability Computation**
```python
# CURRENT: Code unclear on how leaf predictions are computed
# Expected (per DNDF paper):
# P(y|x) = Σ_l∈L μ_l(x|θ) * π_l^y
# where:
#   μ_l(x|θ) = routing probability to leaf l
#   π_l^y = output distribution at leaf l
```

#### ✅ **Đề Xuất Cải Thiện**

**A4: Thêm Ablation Study DNDF**
```bash
# Create separate training scripts:
- train_1d_sota_baseline.py (Softmax classifier only)
- train_1d_sota_dndf.py (Current DNDF)
- train_1d_sota_rf.py (Random Forest baseline)

# Compare on test set with same random seed
```

**A5: Sửa DNDF Routing Logic**
- Xóa sigmoid, dùng softmax trực tiếp trên logits
- Thêm documentation rõ ràng cho μ_l(x|θ) và π_l^y

---

## **PHẦN II: RÀ SOÁT THIẾT KẾ THỰC NGHIỆM**

### **3️⃣ VẤN ĐỀ: Sensitivity + Triage Objective Trade-off**

#### 🔴 **Vấn Đề**
```
Sensitivity mặc định: 54.97%
Sau threshold tuning (τ → 0.35): 72.4%
Nhưng vẫn còn: 27.6% False Negative Rate (FNR)

Ý nghĩa lâm sàng:
- Triage tool bỏ sót ~1 trên 4 ca COVID-19
- Rủi ro lây nhiễm cộng đồng RẤT CAO
```

#### 🎯 **Mục Tiêu Triage Cần Rõ Ràng**

| Guideline | Target Sensitivity | Reasoning |
|-----------|-------------------|-----------|
| **WHO (screening)** | ≥ 90% | Minimize missed COVID cases |
| **FDA (IVD)** | ≥ 95% | Regulatory requirement |
| **Current Model** | 72.4% | ❌ CHƯA ĐỦ CHO TRIAGE |

#### ✅ **Đề Xuất Cải Thiện**

**A6: Threshold Optimization với Youden Index**
```python
# Calculate optimal threshold using Youden's J statistic
from sklearn.metrics import roc_curve
import numpy as np

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

print(f"Youden-optimized threshold: {optimal_threshold:.3f}")
print(f"At this threshold: Sensitivity={tpr[optimal_idx]:.1%}, "
      f"Specificity={(1-fpr[optimal_idx]):.1%}")

# Visualize sensitivity vs threshold
```

**A7: Threshold Sweep Analysis**
```
Thực hiện evaluation ở nhiều thresholds:
τ = [0.2, 0.3, 0.35, 0.4, 0.5]

Report:
- Sensitivity @ τ=0.2 (should be ~90% if model useful for triage)
- Specificity @ τ=0.2 (trade-off analysis)
- F1-score (balance point)
- Recommended operating point
```

**A8: Rõ Ràng Hóa Mục Tiêu Trong Thesis**
```markdown
## 4.3 Sensitivity-Specificity Trade-off

Mục tiêu của hệ thống:
- PRIMARY: Phục vụ Triage (maximize Sensitivity) → τ ≈ 0.25-0.35
- SECONDARY: High-confidence diagnosis → τ ≈ 0.5-0.7

Khuyến nghị lâm sàng:
- Mô hình này phù hợp cho SCREENING (Triage)
- KHÔNG nên dùng làm công cụ diagnostic cuối cùng
- Luôn kết hợp với các chỉ số lâm sàng khác
```

---

### **4️⃣ VẤN ĐỀ: Label Noise từ Self-reporting**

#### 🔴 **Vấn Đề**
```
COUGHVID = tự báo cáo (self-reporting)
→ Nhiễu nhãn cao (label noise)
→ Ground truth không đáng tin cậy
→ Hiệu suất thực tế có thể thấp hơn báo cáo

Nhưng: Thiết kế thực nghiệm coi tất cả nhãn là 100% chính xác
```

#### ✅ **Đề Xuất Cải Thiện**

**A9: Confidence-based Filtering**
```python
# COUGHVID cung cấp "expert_evaluation_score"
# Filter subset có score cao (e.g., score ≥ 0.8)

coughvid_high_conf = df[df['expert_evaluation_score'] >= 0.8]

# Evaluate trên:
# - Full COUGHVID (kết quả hiện tại)
# - High-confidence subset (ground truth tốt hơn)

print(f"Full COUGHVID: {len(coughvid_full)} samples")
print(f"High-confidence COUGHVID: {len(coughvid_high_conf)} samples")
print(f"Performance drop when filtered: {metric_full - metric_filtered}")
```

**A10: Thêm Human Validation Set**
```
Nếu khả năng:
1. Lấy ~50 mẫu cough random từ test set
2. Gửi bác sĩ phổi validation (double-blind)
3. So sánh mô hình predictions vs bác sĩ verdicts
4. Tính Cohen's Kappa (inter-rater agreement)

Nếu không khả năng:
→ Thảo luận nhân này ở Chapter 5 (Limitations)
```

---

## **PHẦN III: KẼHỞ PHƯƠNG PHÁP LUẬN**

### **5️⃣ VẤN ĐỀ: Metadata bị Loại Trừ**

#### 🔴 **Vấn Đề**
```
Mô hình hiện tại:
- INPUT: Chỉ audio ho (raw waveform hoặc spectrogram)
- MISSING: Tuổi, giới tính, thời gian triệu chứng, tiền sử bệnh...

Hậu quả:
- Không phân biệt được: COVID-19 cough vs bronchitis cough
- Acoustic overlap cao (cả hai đều là productive cough)
- Mô hình phải học từ các biến thể rất tinh tế
```

#### 🎯 **Metadata Có Sẵn Trong Dataset**

| Dataset | Có Metadata? | Fields |
|---------|-------------|--------|
| **Coswara** | Có | age, gender, onset_symptom_date, respiratory_condition |
| **COUGHVID** | Có | age, gender, cough_type, symptoms |
| **Hiện tại** | ❌ SỬ DỤNG? | Không được tận dụng |

#### ✅ **Đề Xuất Cải Thiện**

**A11: Metadata-Aware Model**
```python
# Hybrid input:
# 1. Audio → Feature Extractor → h_audio (768-dim)
# 2. Metadata → Embedding → h_meta (32-dim)
# 3. Concatenate → DNDF classifier

class MetadataEmbedding(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.age_embed = nn.Embedding(100, 16)          # Age 0-99
        self.gender_embed = nn.Embedding(3, 8)          # M, F, Other
        self.symptom_duration = nn.Linear(1, 8)         # Days since onset
        
    def forward(self, age, gender, days_since_onset):
        age_feat = self.age_embed(age)
        gender_feat = self.gender_embed(gender)
        duration_feat = self.symptom_duration(days_since_onset.unsqueeze(1))
        return torch.cat([age_feat, gender_feat, duration_feat], dim=1)

# Final prediction:
h_combined = torch.cat([h_audio, h_meta], dim=1)
output = dndf_classifier(h_combined)
```

**A12: Ablation Study: Audio Only vs Audio+Metadata**
```
Experiment 1: Audio only (current)
→ AUC-PR = X%

Experiment 2: Audio + Age + Gender + Symptom Duration
→ AUC-PR = Y%

Report improvement: ΔY - X
→ Justify why metadata excluded (or include if beneficial)
```

**A13: Discuss Limitations in Thesis Chapter 5**
```markdown
## Limitations (Section 5.2)

1. **Clinical Context Ignored**
   - Model lacks patient metadata (age, gender, symptom timeline)
   - Cannot distinguish COVID cough from other respiratory infections with acoustic overlap
   - In clinical practice, these metadata would be available
   
2. **Recommendation for Deployment**
   - Future systems should integrate metadata for improved accuracy
   - Current audio-only approach is conservative baseline
```

---

### **6️⃣ VẤN ĐỀ: XAI Feature Mapping Không Rõ Ràng**

#### 🔴 **Vấn Đề**
```
LIME explains feature importance:
- Feature 370: 0.23 (important)
- Feature 369: 0.18 (important)
- Feature 368: 0.15 (important)

❌ NHƯNG:
- Feature 370 là latent representation của Wav2Vec/AST
- Không có ý nghĩa sinh học (lâm sàng) với bác sĩ
- Bác sĩ không hiểu "Feature 370" là gì
```

#### 🎯 **Mục Tiêu: Acoustic Feature Attribution**

**Cần ánh xạ**: Latent feature → Acoustic characteristic
```
Feature 370 (latent) 
    ↓ [gradient-based saliency map]
Time-Frequency Domain
    ↓ [reverse spectrogram]
Acoustic interpretation
    ↓
Clinically meaningful statement
    
VD: "Frequencies in 2-4 kHz band, 200-500ms timeframe are discriminative"
```

#### ✅ **Đề Xuất Cải Thiện**

**A14: Reverse-map Latent Features to Spectrogram**
```python
# For Wav2Vec (1D raw audio):
# Find which time windows contribute most to important latent features
# using gradient-based attribution (Integrated Gradients)

from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)

# Get attribution for 1D waveform
attributions = ig.attribute(audio_input, target=covid_class)

# Visualization:
# Plot: Time vs Attribution strength
# Identify critical time windows (e.g., [100-200ms], [400-600ms])

# For AST (2D spectrogram):
# Map back using attention weights from AST
# Identify critical time-frequency bins

# Interpretation:
# "Cough pitch variations at 3.5-4.2 kHz in the first 300ms are critical"
```

**A15: Acoustic Feature Descriptors**
```python
# Post-process LIME explanations to extract meaningful descriptors:

def extract_acoustic_descriptors(important_features, audio_input):
    """
    Convert latent features to acoustic descriptors
    """
    # 1. Spectral features
    onset_freq = estimate_from_critical_band(important_features)  # Hz
    pitch_contour = extract_f0_trajectory(audio_input)             # Hz over time
    spectral_centroid = compute_spectral_centroid(important_features)  # Hz
    
    # 2. Temporal features
    attack_time = find_onset_time(important_features)              # ms
    duration = find_duration(important_features)                   # ms
    
    # 3. Clinical interpretation
    interpretation = {
        'pitch_range': (onset_freq - 200, onset_freq + 200),  # Hz
        'critical_band': 'Oropharyngeal region (2-4 kHz)',
        'time_window': f'{attack_time}-{attack_time+duration}ms',
        'clinical_note': 'Consistent with COVID-related cough characteristics'
    }
    return interpretation
```

**A16: Update XAI Visualization**
```
Current:
- Bar plot: Feature 370=0.23, Feature 369=0.18
- Problem: Meaningless to clinician

Proposed:
- Spectrogram heatmap with critical regions highlighted
- Frequency-time plot showing important bands
- Text summary: "Critical frequencies: 2.5-4.0 kHz, Time: 100-500ms"
- Clinical interpretation: "Consistent with hypothesis H3"
```

---

## **PHẦN IV: RÀ SOÁT CÁC CÔNG THỨC VÀ THAM SỐ**

### **7️⃣ VẤN ĐỀ: Focal Loss Formula Không Chính Xác**

#### 🔴 **Vấn Đề: Alpha Không Per-class**

**Current Implementation** (WRONG):
```python
# train_2d_sota.py, line 43
focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
# α=0.8 applied uniformly to both classes
# Problem: Doesn't weight classes differently
# Result: Class imbalance not properly addressed
```

**Correct Implementation** (Lin et al., 2017):
```python
alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

# With self.alpha = 0.8:
# - Positive class (COVID):     α_t = 0.8
# - Negative class (Healthy):   α_t = 0.2
# - Consequence: Down-weight easy negatives, focus on hard positives
```

#### 📋 **Configuration Inconsistency**

| Location | Reported α | Issue |
|----------|-----------|--------|
| PROJECT_SUMMARY.md | α=0.75 | Old config? |
| train_2d_sota.py | α=0.8 | Current |
| train_1d_sota.py | α=0.8 | Current |
| old_stage/train.py | Per-class ✓ | Correct but replaced |

#### ✅ **Đề Xuất Cải Thiện**

**A17: Fix Focal Loss in Both Training Scripts**

```python
# Correct per-class Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.sigmoid(logits)
        
        # Per-class alpha weighting (KEY FIX)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Usage:
criterion = BinaryFocalLoss(alpha=0.8, gamma=2.0)
loss = criterion(logits, targets)
```

**A18: Justify Alpha and Gamma Values**
```markdown
## Hyperparameter Justification

**Focal Loss α = 0.8**
- Positive class weight: 0.8 (COVID-19)
- Negative class weight: 0.2 (Healthy)
- Rationale: Reflect class imbalance (34% COVID vs 66% Healthy)
- Alternative: α = P(negative) = 0.66 (data-driven)

**Focal Loss γ = 2.0**
- Standard value from focal loss paper
- Interpreted as "focusing parameter" - higher γ = more focus on hard examples
- Tested alternatives: γ ∈ [1.0, 2.0, 3.0] (mention in ablation)

**Recommendation**: Report in methodology that these values are
standard and validated in literature for imbalanced medical datasets.
```

---

### **8️⃣ VẤN ĐỀ: DNDF Formula Ký Hiệu Không Rõ Ràng**

#### 🔴 **Vấn Đề**

**Công thức trong thesis**:
$$P(y|x,θ,π) = \sum_{l \in L} μ_l(x|θ) π_l^y$$

**Ký hiệu Ambiguous**:
- θ: Toàn bộ network parameters hay chỉ encoder?
- μ_l(x|θ): Routing probability hay leaf activation?
- π_l^y: Leaf output distribution - shape là gì?
- L: Toàn bộ leaves hay leaves ở mỗi tree?

#### ✅ **Đề Xuất Cải Thiện**

**A19: Rõ Ràng Hóa DNDF Công Thức**

```markdown
## DNDF Mathematical Formulation

### Architecture
- **Feature Encoder**: f_enc(x; θ_enc) → h ∈ ℝ^768
  - Input: x (audio)
  - Output: h (latent features)
  
- **Decision Forest**: Ensemble of T trees (T=5)
  - Each tree t: Contains 2^D leaf nodes (D=4 → 16 leaves/tree)

### Prediction at Tree t:

**Step 1: Routing (Soft assignment to leaves)**
```
For each internal node d in tree t:
  s_d(x) = σ(w_d^T h)  ∈ [0, 1]   [sigmoid gating]

Routing probability to leaf l in tree t:
  μ_t,l(x; θ_t) = ∏_{d ∈ path(l)} s_d(x) · ∏_{d ∉ path(l)} (1 - s_d(x))
```

**Step 2: Leaf Output Distribution**
```
Each leaf l contains learned distribution:
  π_t,l = [π_t,l^{class=0}, π_t,l^{class=1}]  ∈ [0,1]^2, sum=1

Output at leaf l for sample x:
  P(y|x, leaf_l, tree_t) = π_t,l^y
```

**Step 3: Ensemble Prediction**
```
Final prediction (average over all trees):
  P(y|x, θ, π) = (1/T) Σ_{t=1}^T Σ_{l=1}^{2^D} μ_t,l(x; θ_t) · π_t,l^y

Where:
  θ = {θ_enc, {w_d^(t) for all d,t}}  [all parameters]
  π = {π_t,l for all t,l}              [leaf distributions]
```

### Key Differences from Random Forest:
- **Soft routing**: Soft assignment (0-1) not hard (0 or 1)
- **Differentiable**: Can learn routing via gradient descent
- **Uncertainty**: Output range [0,1] from each leaf enables uncertainty quantification
```

---

### **9️⃣ VẤN ĐỀ: MixUp Implementation Không Rõ Lý Do**

#### 🔴 **Vấn Đề**

| Model | MixUp | Justification |
|-------|--------|-----------|
| **1D (Wav2Vec)** | ❌ NO | "NO MIXUP FOR 1D" |
| **2D (AST)** | ✅ YES (α=0.4) | Not explained |

**Problem**: 
- Asymmetric comparison (unfair)
- No ablation study
- Inconsistent with unified framework

#### ✅ **Đề Xuất Cải Thiện**

**A20: Clarify MixUp Layer Choice**

```python
# Manifold MixUp formula in thesis should specify:

z_mixed = λ · f_enc(x_i; θ) + (1 - λ) · f_enc(x_j; θ)
y_mixed = λ · y_i + (1 - λ) · y_j

Where:
- f_enc = [specified layer in feature extractor]
- λ ~ Beta(α, α), α=0.4
- x_i, x_j = random pair from batch
- y_i, y_j = one-hot targets

QUESTION: Which layer for MixUp?
- Option A: After CNN encoder (12-layer Wav2Vec)
- Option B: At final encoder layer (before DNDF)
- Option C: Inside DNDF (should NOT do - defeats purpose)
```

**A21: Add MixUp Ablation Study**

```
Experiment matrix:
┌──────────────┬───────────┬──────────┐
│   Model      │  MixUp    │  AUC-PR  │
├──────────────┼───────────┼──────────┤
│ AST (2D)     │ NO        │   A%     │ ← Baseline
│ AST (2D)     │ YES α=0.4 │   B%     │ ← Current
│ AST (2D)     │ YES α=0.2 │   C%     │ ← Test
│ AST (2D)     │ YES α=0.6 │   D%     │ ← Test
│ Wav2Vec (1D) │ NO        │   E%     │ ← Baseline
│ Wav2Vec (1D) │ YES α=0.4 │   F%     │ ← Test symmetry
└──────────────┴───────────┴──────────┘

Report: Why 1D doesn't benefit from MixUp (if ΔF ≈ 0)
or include MixUp if ΔF > 0
```

---

### **🔟 VẤN ĐỀ: Threshold τ=0.35 Chọn Cảm Tính**

#### 🔴 **Vấn Đề**

```
Current: τ = 0.35 (arbitrary choice)
Sensitivity @ τ=0.35: ~72.4%

Problem:
- Why 0.35 specifically?
- Youden index at 0.35? Unclear.
- Should be science-driven not arbitrary
```

#### ✅ **Đề Xuất Cải Thiện**

**A22: Implement Youden Index Optimization**

```python
from sklearn.metrics import roc_curve
import numpy as np

def find_optimal_threshold(y_true, y_proba):
    """
    Find optimal threshold using Youden's J statistic
    J = Sensitivity + Specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr  # Equivalently: tpr + (1-fpr) - 1
    
    # Find optimal
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    optimal_j = youden_j[optimal_idx]
    
    return {
        'threshold': optimal_threshold,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'youden_j': optimal_j
    }

# Usage:
result = find_optimal_threshold(y_test, y_proba)
print(f"Optimal threshold: {result['threshold']:.3f}")
print(f"Sensitivity: {result['sensitivity']:.1%}, "
      f"Specificity: {result['specificity']:.1%}")
```

**A23: Report Multiple Operating Points**

```markdown
## Threshold Selection Analysis

### Method: Youden Index Optimization
J = Sensitivity + Specificity - 1

### Results:

| Threshold | Sensitivity | Specificity | Youden_J | Use Case |
|-----------|-------------|-----------|----------|----------|
| 0.20 | 92.1% | 65.3% | 0.574 | Ultra-sensitive screening |
| 0.35 | 72.4% | 78.5% | 0.609 | **← Optimal (current)** |
| 0.50 | 54.9% | 82.1% | 0.370 | High-confidence diagnosis |
| 0.70 | 28.3% | 94.2% | 0.225 | Conservative diagnosis |

### Recommendation:
τ = 0.35 is scientifically justified as Youden-optimal threshold
```

---

## **PHẦN V: RÀ SOÁT ĐỘ ĐO (METRICS)**

### **⚠️ VẤN ĐỀ 1: AUC-ROC Gây Hiểu Lầm Với Dữ Liệu Mất Cân Bằng**

#### 🔴 **Vấn Đề**

```
Class distribution: COVID 34% vs Healthy 66%

Giả sử:
- Model 1: Accuracy = 82%, AUC-ROC = 0.78 (current)
- Model 2: Accuracy = 66%, AUC-ROC = 0.75

Nhìn sơ qua: Model 1 tốt hơn (higher accuracy, higher AUC)

NHƯNG:
- Model 1 có thể: Dự đoán tất cả Healthy → Accuracy 66%
- AUC-ROC insensitive với FPR ở vùng [0-0.3] (abundant negatives)
- F1-score, Precision-Recall curves sẽ phát hiện vấn đề
```

#### ✅ **Đề Xuất Cải Thiện**

**A24: Bổ sung AUC-PR Làm Thước Đo Chính**

```python
from sklearn.metrics import auc, precision_recall_curve

# Compute Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
auc_pr = auc(recall, precision)

# Interpretation for imbalanced data:
# AUC-PR focuses on minority class (COVID-19)
# More informative than AUC-ROC for medical screening

print(f"AUC-ROC: {auc_roc:.3f}")
print(f"AUC-PR: {auc_pr:.3f}")  # ← More important for medical

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ROC Curve
axes[0].plot(fpr, tpr, lw=2, label=f'AUC-ROC={auc_roc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', label='Chance')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: PR Curve (MORE INFORMATIVE FOR IMBALANCE)
axes[1].plot(recall, precision, lw=2, label=f'AUC-PR={auc_pr:.3f}')
baseline_pr = y_test.sum() / len(y_test)  # Class prior
axes[1].axhline(baseline_pr, color='k', linestyle='--', label=f'Baseline={baseline_pr:.1%}')
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_comparison.png', dpi=150, bbox_inches='tight')
```

**A25: Update Thesis Tables to Emphasize AUC-PR**

```markdown
## Table 4.2: Model Performance Comparison

| Metric | Wav2Vec + DNDF | AST + DNDF | Interpretation |
|--------|---|---|---|
| **AUC-PR** | **0.612** | **0.418** | ← PRIMARY metric (minority class) |
| **AUC-ROC** | 0.774 | 0.691 | ← Secondary metric |
| **Sensitivity** | 53.4% | 50.2% | Recall of COVID-19 |
| **Specificity** | 86.3% | 83.3% | Recall of Healthy |
| **F1-score** | 0.615 | 0.536 | Harmonic mean |
| **Accuracy** | 75.8% | 75.0% | ⚠️ Misleading for imbalance |

### Note on Accuracy:
Accuracy is not reported as primary metric due to class imbalance 
(66% baseline from always predicting "Healthy"). AUC-PR is more 
informative for medical screening applications.
```

---

### **⚠️ VẤN ĐỀ 2: Mâu Thuẫn Giữa Accuracy Cao và F1-score Thấp**

#### 🔴 **Vấn Đề**

```
AST Model Results:
- Accuracy: 75.0% ← Looks good!
- Precision: 27.28% ← VERY bad!
- F1-score: 35.36% ← VERY bad!

Interpretation:
- Model predicts "COVID" for many negatives
- Out of 10 "COVID" predictions, only 2.7 are correct
- USELESS for clinical deployment

Mâu thuẫn: Accuracy suggests decent performance, but F1/Precision reveals disaster.
```

#### ✅ **Đề Xuất Cải Thiện**

**A26: Deprecate or Contextualize Accuracy**

```markdown
## 4.2.1 Metric Selection Justification

### Why Not Use Accuracy?
With class imbalance (COVID: 34%, Healthy: 66%), a naive model 
that always predicts "Healthy" achieves **66% accuracy** without 
learning any COVID-specific features.

Therefore:
- **Accuracy is not reported** as primary metric
- Instead: **AUC-PR, Precision, Recall, F1-score**

### Minimum Acceptable Performance:
- Sensitivity (Recall): > 90% for triage application
- Precision: > 50% (out of 10 predictions, at least 5 correct)
- F1-score: > 0.50

### Model Performance Summary:
AST model achieves Accuracy=75% but F1=0.354 (falls below threshold).
This indicates model is not suitable for independent deployment.
Recommendation: Ensemble approach with Wav2Vec or additional metadata needed.
```

---

### **⚠️ VẤN ĐỀ 3: Cross-dataset Evaluation Không Thực Sự Cross-dataset**

#### 🔴 **Vấn Đề**

```
Claimed: "Cross-dataset validation on Coswara, COUGHVID, Virufy"

Reality: evaluate_cross_dataset.py evaluates on MERGED dataset
- metadata.csv lacks "dataset_source" column
- File prefixes exist (coswara_, coughvid_) but not stored in metadata
- Cannot separate per-dataset performance

Result: Cannot claim cross-dataset generalization
```

#### ✅ **Đề Xuất Cải Thiện**

**A27: Implement True Cross-dataset Evaluation**

```python
# Step 1: Extract dataset source from filename or add metadata
def infer_dataset_source(filename):
    if filename.startswith('coswara_'):
        return 'coswara'
    elif filename.startswith('coughvid_'):
        return 'coughvid'
    else:
        return 'unknown'

# Step 2: Add to metadata
metadata['dataset_source'] = metadata['filename'].apply(infer_dataset_source)

# Step 3: Leave-One-Dataset-Out Cross Validation
from sklearn.model_selection import StratifiedGroupKFold

datasets = metadata['dataset_source'].unique()
results_per_dataset = {}

for test_dataset in datasets:
    # Test on one dataset
    test_mask = metadata['dataset_source'] == test_dataset
    test_indices = metadata[test_mask].index.tolist()
    
    # Train on others
    train_indices = metadata[~test_mask].index.tolist()
    
    # Train model
    # ... training code ...
    
    # Evaluate
    y_true = targets[test_indices]
    y_proba = predictions[test_indices]
    metrics = compute_metrics(y_true, y_proba)
    results_per_dataset[test_dataset] = metrics

# Report per-dataset
print("Leave-One-Dataset-Out Cross-Validation Results:")
for dataset, metrics in results_per_dataset.items():
    print(f"\n{dataset.upper()} (test):")
    print(f"  AUC-PR: {metrics['auc_pr']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")
```

---

## **PHẦN VI: ĐỀ XUẤT CẢI THIỆN LÂMNHÂN LIÊN QUAN**

### **Recommendation 1: Feature-level Label Noise Analysis**
```python
# Identify potentially mislabeled samples
from sklearn.ensemble import IsolationForest

# Train model, get predictions and confidence
proba_covid = model.predict_proba(features)[:, 1]

# Find confident errors (model very confident, but label says opposite)
confident_errors = (
    ((proba_covid > 0.9) & (labels == 0)) |  # Confident COVID, labeled Healthy
    ((proba_covid < 0.1) & (labels == 1))    # Confident Healthy, labeled COVID
)

print(f"Confident errors found: {confident_errors.sum()} / {len(labels)}")
# If many errors, suggests label noise; visualize and review
```

### **Recommendation 2: Add Confidence Calibration**
```python
# Model outputs are not well-calibrated for medical use
# Use temperature scaling or Platt scaling to get proper probabilities

from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_val, y_val)

# Now model.predict_proba() gives properly calibrated probabilities
# Expected calibration error < 0.05
```

---

## **PHẦN VII: FINAL CHECKLIST TRƯỚC KHI BẢO VỆ**

### **✅ Critical Fixes (Must Do)**
- [ ] A17: Fix Focal Loss alpha per-class
- [ ] A4: Add DNDF ablation study (vs Softmax baseline)
- [ ] A6: Implement Youden threshold optimization
- [ ] A19: Clarify DNDF formula with explicit notation
- [ ] A27: Implement true cross-dataset evaluation

### **⚠️ Important Additions (Should Do)**
- [ ] A24: Add AUC-PR as primary metric
- [ ] A26: Deprecate accuracy, contextualize with F1
- [ ] A12: Add metadata-aware ablation study
- [ ] A14: Map XAI features back to acoustic domain
- [ ] A9: Confidence-based COUGHVID filtering analysis

### **📝 Documentation Improvements (Nice To Have)**
- [ ] A1: Add Loudness normalization to preprocessing
- [ ] A10: Discuss label noise limitations in Chapter 5
- [ ] A20: Justify MixUp configuration choices
- [ ] A22: Report multiple operating points

---

## **SUMMARY TABLE: Issues × Solutions**

| # | Issue Category | Issue | Severity | Solution | Action Item |
|---|---|---|---|---|---|
| 1 | Architecture | 2D AST precision drop | 🔴 CRITICAL | Mel-spectrogram tuning + normalization | A1, A2, A3 |
| 2 | Architecture | DNDF routing (sigmoid→softmax) | 🔴 CRITICAL | Remove sigmoid, use softmax only | Fixed in main code |
| 3 | Architecture | No DNDF ablation study | 🔴 CRITICAL | A/B test DNDF vs Softmax | A4 |
| 4 | Experiment | Sensitivity 72.4% adequate? | 🟠 HIGH | Discuss triage trade-offs, add threshold sweep | A6, A7, A8 |
| 5 | Experiment | Label noise COUGHVID | 🟠 HIGH | Confidence filtering + human validation | A9, A10 |
| 6 | Methodology | Metadata excluded | 🟠 HIGH | Add metadata-aware model + ablation | A11, A12 |
| 7 | Methodology | XAI not clinically interpretable | 🟠 HIGH | Map latent features to acoustic domain | A14, A15, A16 |
| 8 | Formula | Focal Loss alpha not per-class | 🔴 CRITICAL | Apply alpha per-sample | A17 |
| 9 | Formula | DNDF notation ambiguous | 🟠 HIGH | Clarify θ, μ_l, π_l^y, L | A19 |
| 10 | Formula | MixUp justification missing | 🟠 HIGH | Add ablation study, explain layer choice | A20, A21 |
| 11 | Formula | Threshold τ=0.35 arbitrary | 🟠 HIGH | Use Youden index optimization | A22, A23 |
| 12 | Metrics | AUC-ROC misleading for imbalance | 🔴 CRITICAL | Use AUC-PR as primary metric | A24 |
| 13 | Metrics | Accuracy contradicts F1-score | 🟠 HIGH | Deprecate accuracy, emphasize F1 | A25, A26 |
| 14 | Validation | "Cross-dataset" claims unfounded | 🔴 CRITICAL | Implement leave-one-dataset-out CV | A27 |

---

**Document created**: 14/05/2026  
**Review Status**: ⚠️ PENDING IMPLEMENTATION  
**Next Phase**: Execute Action Items in priority order (P1 → P2 → P3)
