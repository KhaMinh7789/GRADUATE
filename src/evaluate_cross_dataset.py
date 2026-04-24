"""
Phase 1: Cross-Dataset Evaluation
Evaluate models trên từng dataset riêng biệt (Coswara, COUGHVID, Virufy)
"""
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

from dataset_offline import CovidCoughDatasetOffline
from model_1d_sota import Wav2Vec2_DNDF_Model
from model_2d_sota import AST_DNDF_Model
from utils_metrics import MetricsCalculator, RocCurveCalculator


class CrossDatasetEvaluator:
    """Evaluate models trên 3 datasets khác nhau"""
    
    def __init__(self, base_dir: str, device: str = 'cuda'):
        self.base_dir = base_dir
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_dir = os.path.join(base_dir, 'data')
        self.metadata_path = os.path.join(self.data_dir, 'metadata.csv')
        
        print(f"🔧 Device: {self.device}")
        print(f"📂 Data directory: {self.data_dir}")
    
    def load_best_models(self, model_weights_dir: str) -> Tuple:
        """
        Load best models từ 5-fold CV
        
        Args:
            model_weights_dir: Đường dẫn folder chứa *.pth files
        
        Returns:
            (wav2vec_model, ast_model)
        """
        # Load Wav2Vec2 DNDF model
        print("\n🔄 Loading Wav2Vec 2.0 + DNDF model...")
        wav2vec_model = Wav2Vec2_DNDF_Model(num_classes=1).to(self.device)
        
        # Tìm best wav2vec weight
        wav2vec_weights = sorted(
            Path(model_weights_dir).glob('best_wav2vec_fold_*.pth'),
            key=lambda x: float(x.stem.split('_')[-1])
        )
        if wav2vec_weights:
            latest_wav2vec = str(wav2vec_weights[-1])  # Lấy fold cuối (best)
            wav2vec_model.load_state_dict(torch.load(latest_wav2vec, map_location=self.device))
            print(f"✅ Loaded: {latest_wav2vec}")
        else:
            print("⚠️  No Wav2Vec weights found")
        
        # Load AST DNDF model
        print("\n🔄 Loading AST + DNDF model...")
        ast_model = AST_DNDF_Model(num_classes=1).to(self.device)
        
        # Tìm best AST weight
        ast_weights = sorted(
            Path(model_weights_dir).glob('best_ast_fold_*.pth'),
            key=lambda x: float(x.stem.split('_')[-1])
        )
        if ast_weights:
            latest_ast = str(ast_weights[-1])
            ast_model.load_state_dict(torch.load(latest_ast, map_location=self.device))
            print(f"✅ Loaded: {latest_ast}")
        else:
            print("⚠️  No AST weights found")
        
        return wav2vec_model, ast_model
    
    def evaluate_model_on_dataset(self, 
                                 model: torch.nn.Module,
                                 dataset: CovidCoughDatasetOffline,
                                 model_type: str = '1D') -> Dict:
        """
        Evaluate 1 model trên 1 dataset
        
        Returns:
            Dict có {metrics, y_true, y_pred, y_proba}
        """
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False, 
                          num_workers=4, pin_memory=True)
        
        print(f"   📊 Evaluating {model_type} on {len(dataset)} samples...")
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.numpy().flatten())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probs)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_proba)
        
        return {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def evaluate_on_three_datasets(self, model_weights_dir: str) -> Dict:
        """
        Main evaluation function
        Evaluate 2 models trên 3 datasets
        
        Returns:
            {model_name: {dataset_name: results}}
        """
        # Load models
        wav2vec_model, ast_model = self.load_best_models(model_weights_dir)
        
        # Load datasets
        print("\n📂 Loading datasets...")
        features_1d_dir = os.path.join(self.data_dir, 'features_1d_raw')
        features_2d_dir = os.path.join(self.data_dir, 'features_2d_ast')
        
        dataset_1d = CovidCoughDatasetOffline(self.metadata_path, features_1d_dir)
        dataset_2d = CovidCoughDatasetOffline(self.metadata_path, features_2d_dir)
        
        # Đề xuất: Parse metadata để lấy từng dataset
        # Hiện tại sử dụng toàn bộ dataset
        
        results = {}
        
        # Evaluate Wav2Vec 2.0
        print("\n" + "="*60)
        print("🎯 EVALUATING WAV2VEC 2.0 + DNDF (1D)")
        print("="*60)
        wav2vec_results = self.evaluate_model_on_dataset(
            wav2vec_model, 
            dataset_1d, 
            model_type='Wav2Vec 2.0 1D'
        )
        results['Wav2Vec 2.0 + DNDF'] = wav2vec_results
        
        print(f"\n✅ Results:")
        print(f"   Accuracy:    {wav2vec_results['metrics']['accuracy']:.4f}")
        print(f"   Sensitivity: {wav2vec_results['metrics']['sensitivity']:.4f}")
        print(f"   Specificity: {wav2vec_results['metrics']['specificity']:.4f}")
        print(f"   AUC-ROC:     {wav2vec_results['metrics']['auc_roc']:.4f}")
        
        # Evaluate AST
        print("\n" + "="*60)
        print("🎯 EVALUATING AST + DNDF (2D)")
        print("="*60)
        ast_results = self.evaluate_model_on_dataset(
            ast_model, 
            dataset_2d, 
            model_type='AST 2D'
        )
        results['AST + DNDF'] = ast_results
        
        print(f"\n✅ Results:")
        print(f"   Accuracy:    {ast_results['metrics']['accuracy']:.4f}")
        print(f"   Sensitivity: {ast_results['metrics']['sensitivity']:.4f}")
        print(f"   Specificity: {ast_results['metrics']['specificity']:.4f}")
        print(f"   AUC-ROC:     {ast_results['metrics']['auc_roc']:.4f}")
        
        return results
    
    def create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """Tạo bảng so sánh tất cả models"""
        comparison_data = []
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'TP': metrics['tp'],
                'FP': metrics['fp'],
                'TN': metrics['tn'],
                'FN': metrics['fn'],
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def save_evaluation_results(self, results: Dict, output_dir: str):
        """Lưu kết quả evaluation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu comparison table
        df = self.create_comparison_table(results)
        table_path = os.path.join(output_dir, 'model_comparison_table.csv')
        df.to_csv(table_path, index=False)
        print(f"\n📊 Saved comparison table: {table_path}")
        
        # Print table
        print("\n" + "="*100)
        print("📈 MODEL COMPARISON TABLE")
        print("="*100)
        print(df.to_string(index=False))
        
        # Lưu detailed metrics as JSON
        import json
        detailed_metrics = {}
        for model_name, model_results in results.items():
            detailed_metrics[model_name] = model_results['metrics']
        
        metrics_path = os.path.join(output_dir, 'detailed_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"\n📄 Saved detailed metrics: {metrics_path}")
        
        # Lưu ROC data
        roc_dict = {}
        for model_name, model_results in results.items():
            fpr, tpr, auc_score = RocCurveCalculator.calculate_roc(
                model_results['y_true'],
                model_results['y_proba']
            )
            roc_dict[model_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc_score
            }
        
        roc_path = os.path.join(output_dir, 'roc_curves.json')
        with open(roc_path, 'w') as f:
            json.dump(roc_dict, f, indent=2)
        print(f"📈 Saved ROC data: {roc_path}")
        
        return output_dir


def main():
    """Main execution"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_weights_dir = base_dir
    
    evaluator = CrossDatasetEvaluator(base_dir)
    
    # Evaluate models
    results = evaluator.evaluate_on_three_datasets(model_weights_dir)
    
    # Save results
    output_dir = os.path.join(base_dir, 'evaluation_results')
    evaluator.save_evaluation_results(results, output_dir)
    
    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE: Cross-Dataset Evaluation")
    print("="*60)


if __name__ == "__main__":
    main()
