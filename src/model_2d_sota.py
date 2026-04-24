import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel

# ==========================================
# BỘ PHÂN LOẠI: DNDF (Đã sửa chuẩn toán học)
# ==========================================
class NeuralDecisionTree(nn.Module):
    def __init__(self, depth, feature_dim, num_classes=1):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.decision_nodes = nn.Linear(feature_dim, self.num_leaves, bias=False)
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, num_classes))

    def forward(self, x):
        routing_probs = torch.sigmoid(self.decision_nodes(x))
        leaf_probs = F.softmax(routing_probs, dim=1)
        leaf_preds = torch.sigmoid(self.leaves)
        out = torch.matmul(leaf_probs, leaf_preds)
        return out

class DNDF(nn.Module):
    def __init__(self, num_trees, depth, feature_dim, num_classes=1):
        super(DNDF, self).__init__()
        self.trees = nn.ModuleList([
            NeuralDecisionTree(depth, feature_dim, num_classes) for _ in range(num_trees)
        ])

    def forward(self, x):
        tree_outputs = [tree(x) for tree in self.trees]
        return torch.mean(torch.stack(tree_outputs), dim=0)

# ==========================================
# MÔ HÌNH HOÀN CHỈNH: AST + DNDF
# ==========================================
class AST_DNDF_Model(nn.Module):
    def __init__(self, num_classes=1):
        super(AST_DNDF_Model, self).__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_dim = 768 
        self.dndf = DNDF(num_trees=5, depth=4, feature_dim=self.feature_dim, num_classes=num_classes)

    def forward(self, x):
        # 1. Bỏ chiều Kênh (Channel = 1) -> Shape từ [Batch, 1, 128, 1024] thành [Batch, 128, 1024]
        x = x.squeeze(1)
        
        # 2. Đảo chiều Tần số (128) và Thời gian (1024) -> Shape chuẩn: [Batch, 1024, 128]
        x = x.transpose(1, 2)
        
        # 3. Đưa vào lõi AST an toàn
        outputs = self.ast(x)
        features = outputs.last_hidden_state.mean(dim=1)
        
        # 4. Qua Rừng quyết định
        out = self.dndf(features)
        return out