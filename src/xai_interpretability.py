"""
Phase 2: XAI - Explainable AI using LIME
Provide interpretability to model predictions
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

from dataset_offline import CovidCoughDatasetOffline
from model_1d_sota import Wav2Vec2_DNDF_Model
from model_2d_sota import AST_DNDF_Model


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) cho audio models
    Giải thích dự đoán model ở cấp độ local
    """
    
    def __init__(self, model: torch.nn.Module, 
                 device: torch.device,
                 feature_type: str = '1D'):
        """
        Args:
            model: Trained model
            device: GPU/CPU device
            feature_type: '1D' hoặc '2D'
        """
        self.model = model
        self.device = device
        self.feature_type = feature_type
        self.model.eval()
    
    def get_prediction_with_features(self, feature: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Get model prediction và extract intermediate features
        
        Returns:
            (prediction_prob, extracted_features)
        """
        with torch.no_grad():
            feature = feature.unsqueeze(0).to(self.device)  # Add batch dim
            
            if self.feature_type == '1D':
                # Wav2Vec2
                outputs = self.model.wav2vec2(feature)
                hidden_states = outputs.last_hidden_state
                features = hidden_states.mean(dim=1).cpu().numpy()  # [1, 768]
                
            else:  # 2D
                # AST
                feature = feature.squeeze(1)  # Remove channel dim
                feature = feature.transpose(1, 2)  # [1, 1024, 128]
                outputs = self.model.ast(feature)
                hidden_states = outputs.last_hidden_state
                features = hidden_states.mean(dim=1).cpu().numpy()  # [1, 768]
            
            # Get final prediction
            prediction = self.model(feature.to(self.device) if self.feature_type == '1D' 
                                   else feature.unsqueeze(1).to(self.device))
            prediction = prediction.item()
        
        return prediction, features.flatten()
    
    def perturb_features(self, features: np.ndarray, 
                        num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb features theo LIME logic
        
        Returns:
            (perturbed_features, perturbation_mask)
        """
        # Tạo random perturbation masks
        perturbation_masks = np.random.binomial(1, 0.5, size=(num_samples, len(features)))
        
        # Perturb features dựa trên mask
        perturbed_features = perturbation_masks * features
        
        return perturbed_features, perturbation_masks
    
    def explain_prediction(self, feature: torch.Tensor, 
                          num_samples: int = 50,
                          label_names: List[str] = ['Healthy', 'COVID-19']) -> Dict:
        """
        Generate LIME explanation cho 1 prediction
        
        Returns:
            {prediction, explanation_weights, feature_importance}
        """
        # Get base prediction
        pred_prob, extracted_features = self.get_prediction_with_features(feature)
        pred_class = int(pred_prob > 0.5)
        
        print(f"\n🔍 LIME Explanation")
        print(f"   Prediction: {label_names[pred_class]} (prob: {pred_prob:.4f})")
        
        # Generate perturbations
        perturbed, masks = self.perturb_features(extracted_features, num_samples)
        
        # Get predictions for perturbed features
        perturbed_preds = []
        for perturbed_feature in perturbed:
            # Reconstruct the feature for model
            reconstructed = perturbed_feature
            
            # For LIME, we use a simple linear model to approximate
            # the model's behavior locally
            with torch.no_grad():
                feat_tensor = torch.from_numpy(reconstructed).float().unsqueeze(0).to(self.device)
                # Use DNDF for local prediction
                if self.feature_type == '1D':
                    local_pred = self.model.dndf(feat_tensor).item()
                else:
                    local_pred = self.model.dndf(feat_tensor).item()
                perturbed_preds.append(local_pred)
        
        perturbed_preds = np.array(perturbed_preds)
        
        # Calculate feature importance using linear approximation
        # Distance-based weights
        distances = np.linalg.norm(masks - 1, axis=1)
        weights = np.exp(-distances**2 / 2)
        
        # Fit linear model to get feature importance
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(masks, perturbed_preds, sample_weight=weights)
        
        feature_importance = lr.coef_
        
        # Get top features
        top_indices = np.argsort(np.abs(feature_importance))[-10:][::-1]
        
        return {
            'prediction': pred_prob,
            'predicted_class': pred_class,
            'predicted_label': label_names[pred_class],
            'feature_importance': feature_importance,
            'top_feature_indices': top_indices,
            'top_feature_values': feature_importance[top_indices],
            'extracted_features': extracted_features,
        }
    
    def visualize_explanation(self, explanation: Dict, 
                             save_path: str = None):
        """Visualize LIME explanation"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Top features
        ax = axes[0]
        top_indices = explanation['top_feature_indices']
        top_values = explanation['top_feature_values']
        
        colors = ['green' if v > 0 else 'red' for v in top_values]
        ax.barh(range(len(top_values)), top_values, color=colors)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels([f"Feature {i}" for i in top_indices])
        ax.set_xlabel('Importance Weight')
        ax.set_title(f"Top 10 Features (Prediction: {explanation['predicted_label']})")
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Feature importance distribution
        ax = axes[1]
        all_importance = explanation['feature_importance']
        ax.hist(all_importance, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='Zero')
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Importance Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visualization: {save_path}")
        
        plt.close()


class InterpretabilityAnalysis:
    """Phân tích interpretability cho nhiều samples"""
    
    def __init__(self, base_dir: str, device_str: str = 'cuda'):
        self.base_dir = base_dir
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.data_dir = os.path.join(base_dir, 'data')
        self.metadata_path = os.path.join(self.data_dir, 'metadata.csv')
    
    def load_models(self, model_weights_dir: str) -> Tuple:
        """Load trained models"""
        print("🔄 Loading models for XAI analysis...")
        
        wav2vec_model = Wav2Vec2_DNDF_Model(num_classes=1).to(self.device)
        ast_model = AST_DNDF_Model(num_classes=1).to(self.device)
        
        from pathlib import Path
        
        # Load Wav2Vec
        wav2vec_weights = sorted(
            Path(model_weights_dir).glob('best_wav2vec_fold_*.pth'),
            key=lambda x: float(x.stem.split('_')[-1])
        )
        if wav2vec_weights:
            wav2vec_model.load_state_dict(
                torch.load(str(wav2vec_weights[-1]), map_location=self.device)
            )
            print(f"✅ Loaded Wav2Vec: {wav2vec_weights[-1].name}")
        
        # Load AST
        ast_weights = sorted(
            Path(model_weights_dir).glob('best_ast_fold_*.pth'),
            key=lambda x: float(x.stem.split('_')[-1])
        )
        if ast_weights:
            ast_model.load_state_dict(
                torch.load(str(ast_weights[-1]), map_location=self.device)
            )
            print(f"✅ Loaded AST: {ast_weights[-1].name}")
        
        return wav2vec_model, ast_model
    
    def analyze_predictions(self, num_samples: int = 5):
        """Analyze interpretability cho một số samples"""
        
        wav2vec_model, ast_model = self.load_models(self.base_dir)
        
        # Load datasets
        features_1d_dir = os.path.join(self.data_dir, 'features_1d_raw')
        features_2d_dir = os.path.join(self.data_dir, 'features_2d_ast')
        
        dataset_1d = CovidCoughDatasetOffline(self.metadata_path, features_1d_dir)
        dataset_2d = CovidCoughDatasetOffline(self.metadata_path, features_2d_dir)
        
        # Create LIME explainers
        lime_1d = LIMEExplainer(wav2vec_model, self.device, feature_type='1D')
        lime_2d = LIMEExplainer(ast_model, self.device, feature_type='2D')
        
        # Sample random indices
        indices = np.random.choice(len(dataset_1d), min(num_samples, len(dataset_1d)), 
                                  replace=False)
        
        output_dir = os.path.join(self.base_dir, 'xai_explanations')
        os.makedirs(output_dir, exist_ok=True)
        
        all_explanations = {'1D': [], '2D': []}
        
        print("\n" + "="*60)
        print("🎯 GENERATING LIME EXPLANATIONS")
        print("="*60)
        
        for i, idx in enumerate(indices):
            print(f"\n📊 Sample {i+1}/{len(indices)}")
            
            # Get features
            feature_1d, label = dataset_1d[idx]
            feature_2d, _ = dataset_2d[idx]
            
            label_names = ['Healthy', 'COVID-19']
            
            # Explain 1D
            print(f"   🔍 Explaining Wav2Vec 2.0...")
            exp_1d = lime_1d.explain_prediction(feature_1d, 
                                               label_names=label_names)
            all_explanations['1D'].append(exp_1d)
            
            save_path_1d = os.path.join(output_dir, 
                                       f'sample_{idx}_wav2vec_explanation.png')
            lime_1d.visualize_explanation(exp_1d, save_path_1d)
            
            # Explain 2D
            print(f"   🔍 Explaining AST...")
            exp_2d = lime_2d.explain_prediction(feature_2d,
                                               label_names=label_names)
            all_explanations['2D'].append(exp_2d)
            
            save_path_2d = os.path.join(output_dir,
                                       f'sample_{idx}_ast_explanation.png')
            lime_2d.visualize_explanation(exp_2d, save_path_2d)
            
            print(f"   True Label: {label_names[int(label)]}")
            print(f"   Wav2Vec Prediction: {exp_1d['predicted_label']} "
                 f"(prob: {exp_1d['prediction']:.4f})")
            print(f"   AST Prediction: {exp_2d['predicted_label']} "
                 f"(prob: {exp_2d['prediction']:.4f})")
        
        # Save summary
        summary_path = os.path.join(output_dir, 'xai_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("XAI LIME EXPLANATIONS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for model_type, exps in all_explanations.items():
                f.write(f"\n{model_type} Model:\n")
                f.write(f"  Number of samples analyzed: {len(exps)}\n")
                f.write(f"  Average confidence: "
                       f"{np.mean([e['prediction'] for e in exps]):.4f}\n")
        
        print(f"\n✅ Saved XAI analysis to: {output_dir}")
        return all_explanations


def main():
    """Main execution"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    analyzer = InterpretabilityAnalysis(base_dir)
    explanations = analyzer.analyze_predictions(num_samples=5)
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE: XAI Interpretability (LIME)")
    print("="*60)


if __name__ == "__main__":
    main()
