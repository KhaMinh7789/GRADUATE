import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. BỘ PHÂN LOẠI: DNDF
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
        leaf_probs = torch.sigmoid(self.leaves)
        return torch.matmul(routing_probs, leaf_probs)

class DNDF(nn.Module):
    def __init__(self, num_trees, depth, feature_dim, num_classes):
        super(DNDF, self).__init__()
        self.trees = nn.ModuleList([
            NeuralDecisionTree(depth, feature_dim, num_classes) for _ in range(num_trees)
        ])

    def forward(self, x):
        tree_outputs = [tree(x) for tree in self.trees]
        return torch.mean(torch.stack(tree_outputs), dim=0)

# 2. MÔ HÌNH HOÀN CHỈNH: 1D-CNN + DNDF
class HybridModel1D(nn.Module):
    def __init__(self, num_classes=1):
        super(HybridModel1D, self).__init__()
        
        # Mạng 1D-CNN chuyên xử lý chuỗi tín hiệu thời gian
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 128
        
        # DNDF tương tự nhánh 2D
        self.dndf = DNDF(num_trees=5, depth=4, feature_dim=self.feature_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        features = torch.flatten(x, 1) # Output: [Batch, 128]
        out = self.dndf(features)
        return out