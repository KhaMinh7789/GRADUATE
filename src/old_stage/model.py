import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. BỘ PHÂN LOẠI: DNDF (Neural Decision Forest)
# ==========================================
class NeuralDecisionTree(nn.Module):
    def __init__(self, depth, feature_dim, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        self.decision_nodes = nn.Linear(feature_dim, self.num_leaves, bias=False)
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, num_classes))

    def forward(self, x):
        routing_probs = torch.sigmoid(self.decision_nodes(x))
        routing_probs = F.softmax(routing_probs, dim=1)
        
        # Dùng Sigmoid cho bài toán phân loại nhị phân
        leaf_probs = torch.sigmoid(self.leaves)
        
        out = torch.matmul(routing_probs, leaf_probs)
        return out

class DNDF(nn.Module):
    def __init__(self, num_trees, depth, feature_dim, num_classes):
        super(DNDF, self).__init__()
        self.trees = nn.ModuleList([
            NeuralDecisionTree(depth, feature_dim, num_classes) for _ in range(num_trees)
        ])

    def forward(self, x):
        tree_outputs = [tree(x) for tree in self.trees]
        forest_out = torch.mean(torch.stack(tree_outputs), dim=0)
        return forest_out

# ==========================================
# 2. MÔ HÌNH HOÀN CHỈNH: RESNET18 + DNDF
# ==========================================
class HybridModel2D(nn.Module):
    def __init__(self, num_classes=1):
        super(HybridModel2D, self).__init__()
        
        # 1. Tải bộ não ResNet18 đã học trước từ Google (Pre-trained)
        resnet = models.resnet18(pretrained=True)
        
        # 2. Cắt bỏ lớp phân loại cuối cùng của ResNet, chỉ lấy phần trích xuất đặc trưng
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18 luôn trả về vector đặc trưng có kích thước 512
        self.feature_dim = 512 
        
        # 3. Chuyển vector 512 này vào Rừng quyết định (DNDF) của bạn
        # Tăng độ sâu của cây lên 4 để chứa được nhiều quy luật phức tạp hơn
        self.dndf = DNDF(num_trees=5, depth=4, feature_dim=self.feature_dim, num_classes=num_classes)

    def forward(self, x):
        # Ảo thuật: Biến ảnh phổ 1 kênh (Trắng đen) thành 3 kênh (RGB) 
        # để đánh lừa ResNet18 (vì ResNet được train trên ảnh màu)
        x = x.repeat(1, 3, 1, 1)
        
        # Trích xuất đặc trưng sâu
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # Output: [Batch, 512]
        
        # Phân loại bằng DNDF
        out = self.dndf(features)
        return out

if __name__ == "__main__":
    print("⏳ Đang test mô hình Hybrid ResNet18 + DNDF...")
    model = HybridModel2D(num_classes=1)
    dummy_input = torch.randn(4, 1, 64, 157)
    output = model(dummy_input)
    print("✅ Cấu trúc mô hình hoạt động hoàn hảo!")
    print(f"-> Kích thước Output: {output.shape}")