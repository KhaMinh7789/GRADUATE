import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel

# ==========================================
# BỘ PHÂN LOẠI: DNDF (Corrected - v2 aligned with model_1d_sota.py)
# FIX: Removed erroneous sigmoid→softmax routing.
# CORRECT: softmax applied directly to raw logits (standard DNDF formulation)
# Ref: P(y|x,θ,π) = Σ_{l∈L} μ_l(x|θ) · π_l^y
#   where μ_l = leaf routing probs (via softmax over logits)
#         π_l^y = learned leaf class distribution
# ==========================================
class NeuralDecisionTree(nn.Module):
    def __init__(self, depth, feature_dim, num_classes=1):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth  # D=4 -> 16 leaves per tree
        # Decision node: maps feature (θ_enc output) to leaf routing logits
        self.decision_nodes = nn.Linear(feature_dim, self.num_leaves, bias=False)
        # Leaf parameters: π_l^y ∈ [0,1] learned class distribution per leaf
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, num_classes))

    def forward(self, x):
        # FIXED (v2): Use softmax DIRECTLY on raw logits (not sigmoid first)
        # routing_logits: raw scores from linear projection of latent features θ
        routing_logits = self.decision_nodes(x)       # [Batch, num_leaves]
        # μ_l(x|θ): soft routing probability to each leaf (sums to 1)
        leaf_probs = F.softmax(routing_logits, dim=1) # [Batch, num_leaves]
        # π_l^y: learned output distribution at each leaf l
        leaf_preds = torch.sigmoid(self.leaves)        # [num_leaves, 1]
        # P(y|x,θ,π) = Σ_l μ_l(x|θ) · π_l^y (weighted sum over all leaves)
        out = torch.matmul(leaf_probs, leaf_preds)     # [Batch, 1]
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
        
        # Multi-scale Feature Fusion
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=7, padding=3)
        self.feature_dim = 256 * 3
        
        self.dndf = DNDF(num_trees=5, depth=4, feature_dim=self.feature_dim, num_classes=num_classes)

    def forward(self, x):
        # 1. Bỏ chiều Kênh (Channel = 1) -> Shape từ [Batch, 1, 128, 1024] thành [Batch, 128, 1024]
        x = x.squeeze(1)
        
        # 2. Đảo chiều Tần số (128) và Thời gian (1024) -> Shape chuẩn: [Batch, 1024, 128]
        x = x.transpose(1, 2)
        
        # 3. Đưa vào lõi AST an toàn
        outputs = self.ast(x)
        # outputs.last_hidden_state: [Batch, SeqLen, 768]
        
        # Transpose cho Conv1d: [Batch, 768, SeqLen]
        hidden_states = outputs.last_hidden_state.transpose(1, 2)
        
        # 4. Multi-scale Fusion (trích xuất thông tin ở nhiều cấp độ phân giải)
        feat1 = F.relu(self.conv1(hidden_states)).mean(dim=2)  # [Batch, 256]
        feat2 = F.relu(self.conv2(hidden_states)).mean(dim=2)  # [Batch, 256]
        feat3 = F.relu(self.conv3(hidden_states)).mean(dim=2)  # [Batch, 256]
        
        fused_features = torch.cat([feat1, feat2, feat3], dim=1)  # [Batch, 768]
        
        # 5. Qua Rừng quyết định
        out = self.dndf(fused_features)
        return out