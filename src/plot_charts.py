import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Cấu hình font và style chuẩn học thuật
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_figure_4_3_1():
    print("Đang vẽ Hình 4.3.1: So sánh AUC 5 Fold...")
    folds = np.arange(1, 6)
    
    # Dữ liệu trích xuất từ Log
    auc_1d = [0.7620, 0.7266, 0.7818, 0.7420, 0.7289]
    auc_2d = [0.6953, 0.6946, 0.6801, 0.6984, 0.6931]
    
    mean_1d = np.mean(auc_1d)
    mean_2d = np.mean(auc_2d)

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    
    # Vẽ cột
    bars1 = ax.bar(folds - bar_width/2, auc_1d, bar_width, label='1D (Wav2Vec 2.0 + DNDF)', color='#2ca02c', edgecolor='black')
    bars2 = ax.bar(folds + bar_width/2, auc_2d, bar_width, label='2D (AST + DNDF)', color='#1f77b4', edgecolor='black')
    
    # Thêm đường trung bình (Mean)
    ax.axhline(mean_1d, color='green', linestyle='--', linewidth=1.5, label=f'1D Mean: {mean_1d:.4f}')
    ax.axhline(mean_2d, color='blue', linestyle='--', linewidth=1.5, label=f'2D Mean: {mean_2d:.4f}')

    # Trang trí biểu đồ
    ax.set_xlabel('Cross-Validation Folds', fontweight='bold')
    ax.set_ylabel('AUC Score', fontweight='bold')
    ax.set_title('Figure 4.3.1: AUC Comparison Across 5 Folds (1D vs 2D)', fontweight='bold', pad=15)
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {i}' for i in folds])
    ax.set_ylim(0.60, 0.85) # Cắt trục Y để thấy rõ sự khác biệt
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hiển thị số trên đỉnh cột
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('Figure_4.3.1_AUC_Comparison.png', dpi=300) # Lưu ảnh HD
    print("✔️ Đã lưu Figure_4.3.1_AUC_Comparison.png")

def plot_figure_4_3_2(log_file_path):
    print("Đang đọc Log và vẽ Hình 4.3.2: Learning Curves...")
    if not os.path.exists(log_file_path):
        print(f"Lỗi: Không tìm thấy file {log_file_path}")
        return

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Cắt lấy nội dung của Fold 3 (Fold tốt nhất của 1D)
    fold_parts = re.split(r'={10,}\s*VÒNG LẶP FOLD \d/5\s*={10,}', content)
    fold_3_content = fold_parts[3] 

    # Regex trích xuất Epoch, Train Loss và Val AUC
    pattern = r'🟢 Epoch (\d+)/15 \| Train Loss \(TB\): ([\d.]+) .*?Metrics -> .*? AUC: ([\d.]+)'
    matches = re.findall(pattern, fold_3_content, re.DOTALL)

    epochs = []
    train_losses = []
    val_aucs = []

    for match in matches:
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
        val_aucs.append(float(match[2]))

    # Vẽ biểu đồ 2 trục Y
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Trục Y thứ nhất cho Loss
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Train Loss', color=color1, fontweight='bold')
    ax1.plot(epochs, train_losses, marker='o', color=color1, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Trục Y thứ hai cho AUC
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Validation AUC', color=color2, fontweight='bold')
    ax2.plot(epochs, val_aucs, marker='s', color=color2, linewidth=2, label='Validation AUC')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Trang trí
    plt.title('Figure 4.3.2: Learning Curves of the Best Model (1D - Fold 3)', fontweight='bold', pad=15)
    fig.tight_layout()
    
    # Gộp legend của 2 trục
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.savefig('Figure_4.3.2_Learning_Curves.png', dpi=300) # Lưu ảnh HD
    print("✔️ Đã lưu Figure_4.3.2_Learning_Curves.png")

if __name__ == "__main__":
    # 1. Vẽ Hình 4.3.1 (Dữ liệu đã được gán sẵn trong hàm)
    plot_figure_4_3_1()
    
    # 2. Khai báo đường dẫn tuyệt đối trên máy của bạn (Nhớ có chữ 'r' ở trước)
    log_1d = r"D:\GROWTH\GRADUTION_MASTER\LOG-Wav2Vec 2.0 (1D Raw Audio) + DNDF.txt"
    log_2d = r"D:\GROWTH\GRADUTION_MASTER\LOG-Audio Spectrogram Transformer (AST) + DNDF.txt"
    
    # 3. Vẽ Hình 4.3.2 (Đang ưu tiên vẽ cho mô hình 1D vì nó là SOTA tốt nhất)
    plot_figure_4_3_2(log_1d)