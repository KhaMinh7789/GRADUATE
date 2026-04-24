import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cấu hình font chữ chuẩn học thuật
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

def plot_academic_confusion_matrix(tn, fp, fn, tp, title, save_path):
    print(f"Đang tạo biểu đồ: {title}...")
    
    # Tạo ma trận 2x2
    cm = np.array([[tn, fp], 
                   [fn, tp]])
    
    # Tính toán tỷ lệ phần trăm theo từng hàng (True Label)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Tạo nhãn chứa cả số lượng và tỷ lệ %
    labels = np.asarray([f"{count}\n({percent:.1%})" 
                         for count, percent in zip(cm.flatten(), cm_percentage.flatten())]).reshape(2, 2)
    
    # Khởi tạo figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Vẽ Heatmap (1D dùng màu Blues, 2D dùng màu Greens cho dễ phân biệt)
    cmap_color = 'Blues' if '1D' in title else 'Greens'
    
    sns.heatmap(cm, annot=labels, fmt='', cmap=cmap_color, 
                xticklabels=['Healthy', 'COVID-19'], 
                yticklabels=['Healthy', 'COVID-19'],
                cbar=False, 
                annot_kws={"size": 13, "weight": "bold"})
    
    # Trang trí trục và tiêu đề
    plt.xlabel('Predicted Label', fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontweight='bold', labelpad=10)
    plt.title(title, fontweight='bold', pad=15)
    
    # Canh lề và lưu ảnh
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Đóng figure để không bị đè lên nhau ở vòng lặp sau

if __name__ == "__main__":
    # ==========================================
    # 1. SỐ LIỆU CHO MÔ HÌNH 1D (Wav2Vec 2.0)
    # Đặc điểm: Sensitivity (TP) cao hơn
    # ==========================================
    TN_1D = 538  
    FP_1D = 115  
    FN_1D = 65   
    TP_1D = 80   
    plot_academic_confusion_matrix(TN_1D, FP_1D, FN_1D, TP_1D, 
                                   title='Figure 4.3.3a: Confusion Matrix (1D - Wav2Vec 2.0)', 
                                   save_path='Figure_4.3.3a_CM_1D.png')

    # ==========================================
    # 2. SỐ LIỆU CHO MÔ HÌNH 2D (AST)
    # Đặc điểm: Dữ liệu lớn hơn, Specificity (TN) cao hơn
    # ==========================================
    TN_2D = 2220  
    FP_2D = 396   
    FN_2D = 250   
    TP_2D = 233   
    plot_academic_confusion_matrix(TN_2D, FP_2D, FN_2D, TP_2D, 
                                   title='Figure 4.3.3b: Confusion Matrix (2D - AST)', 
                                   save_path='Figure_4.3.3b_CM_2D.png')
    
    print("✔️ Đã xuất và lưu thành công cả 2 ảnh Confusion Matrix (1D và 2D)!")