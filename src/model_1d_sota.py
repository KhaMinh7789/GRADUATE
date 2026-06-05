import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

# ==========================================
# BỘ PHÂN LOẠI: DNDF (Đã sửa chuẩn toán học)
# ==========================================
class NeuralDecisionTree(nn.Module):
    def __init__(self, depth, feature_dim, num_classes=1, metadata_dim=0):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        # Thêm metadata_dim vào input của decision_nodes (Routing Nodes)
        self.decision_nodes = nn.Linear(feature_dim + metadata_dim, self.num_leaves, bias=False)
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, num_classes))

    def forward(self, x, metadata=None):
        # FIXED: Use softmax directly on logits (not sigmoid first)
        # Lỗ hổng thiếu Metadata: nhúng các đặc trưng Metadata vào các nút định tuyến
        if metadata is not None:
            routing_input = torch.cat([x, metadata], dim=1)
        else:
            routing_input = x
            
        routing_logits = self.decision_nodes(routing_input)  # [Batch, num_leaves]
        leaf_probs = F.softmax(routing_logits, dim=1)  # [Batch, num_leaves]
        
        # Leaf output predictions (per-leaf class probabilities)
        leaf_preds = torch.sigmoid(self.leaves)  # [num_leaves, 1]
        
        # Final prediction = weighted sum of leaf predictions
        out = torch.matmul(leaf_probs, leaf_preds)  # [Batch, 1]
        return out

class DNDF(nn.Module):
    def __init__(self, num_trees, depth, feature_dim, num_classes=1, metadata_dim=0):
        super(DNDF, self).__init__()
        self.trees = nn.ModuleList([
            NeuralDecisionTree(depth, feature_dim, num_classes, metadata_dim) for _ in range(num_trees)
        ])

    def forward(self, x, metadata=None):
        tree_outputs = [tree(x, metadata) for tree in self.trees]
        return torch.mean(torch.stack(tree_outputs), dim=0)

# ==========================================
# MÔ HÌNH HOÀN CHỈNH: Wav2Vec 2.0 + DNDF
# ==========================================
class Wav2Vec2_DNDF_Model(nn.Module):
    def __init__(self, num_classes=1, metadata_dim=0):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Đóng băng Feature Extractor (CNN)
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Đóng băng các lớp Encoder đầu tiên, MỞ KHÓA 2 lớp cuối (10 và 11)
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if not any(f"layers.{i}." in name for i in range(10, 12)):
                param.requires_grad = False

        self.dndf = DNDF(num_trees=5, depth=4, feature_dim=768, num_classes=num_classes, metadata_dim=metadata_dim)

    def forward(self, x, metadata=None):
        outputs = self.wav2vec2(x)
        hidden_states = outputs.last_hidden_state
        features = hidden_states.mean(dim=1) # Shape: [Batch, 768]
        out = self.dndf(features, metadata)
        return out