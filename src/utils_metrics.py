"""
Utility functions cho tính toán và lưu trữ metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report
)
from typing import Dict, Tuple, List
import json


class MetricsCalculator:
    """Tính toán comprehensive metrics cho binary classification"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             y_proba: np.ndarray) -> Dict:
        """
        Tính toán tất cả metrics
        
        Args:
            y_true: Ground truth labels [0, 1]
            y_pred: Binary predictions [0, 1]
            y_proba: Probability predictions (0-1)
        
        Returns:
            Dict chứa tất cả metrics
        """
        # Flatten để đảm bảo shape đúng
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_proba = np.array(y_proba).flatten()
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred, zero_division=0)  # True Positive Rate (TPR)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (TNR)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC
        try:
            auc_score = roc_auc_score(y_true, y_proba)
        except:
            auc_score = 0.0
        
        # Specificity & Sensitivity
        metrics = {
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1_score': f1,
            'auc_roc': auc_score,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }
        
        return metrics
    
    @staticmethod
    def create_metrics_table(results_dict: Dict) -> pd.DataFrame:
        """
        Tạo bảng metrics từ dictionary kết quả
        
        Args:
            results_dict: Dict có format {dataset_name: metrics}
        
        Returns:
            DataFrame với tất cả metrics
        """
        table_data = []
        
        for dataset_name, metrics in results_dict.items():
            row = {
                'Dataset': dataset_name,
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
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    @staticmethod
    def save_metrics_json(metrics_dict: Dict, output_path: str):
        """Lưu metrics dưới dạng JSON"""
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✅ Lưu metrics tại: {output_path}")
    
    @staticmethod
    def format_metrics_for_paper(metrics: Dict) -> str:
        """Định dạng metrics cho paper (1 dòng)"""
        return (
            f"Accuracy: {metrics['accuracy']:.2%} | "
            f"Sensitivity: {metrics['sensitivity']:.2%} | "
            f"Specificity: {metrics['specificity']:.2%} | "
            f"AUC: {metrics['auc_roc']:.4f}"
        )


class RocCurveCalculator:
    """Tính toán ROC curve"""
    
    @staticmethod
    def calculate_roc(y_true: np.ndarray, 
                     y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Tính FPR, TPR, AUC cho ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score
    
    @staticmethod
    def get_roc_for_multiple_models(results_dict: Dict) -> Dict:
        """
        Tính ROC curves cho nhiều models
        
        Args:
            results_dict: {model_name: {dataset_name: {y_true, y_proba}}}
        """
        roc_dict = {}
        
        for model_name, datasets in results_dict.items():
            roc_dict[model_name] = {}
            for dataset_name, data in datasets.items():
                fpr, tpr, auc_score = RocCurveCalculator.calculate_roc(
                    data['y_true'], 
                    data['y_proba']
                )
                roc_dict[model_name][dataset_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc_score
                }
        
        return roc_dict


class ConfusionMatrixCalculator:
    """Helper cho confusion matrix"""
    
    @staticmethod
    def get_cm_values(y_true: np.ndarray, 
                     y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Lấy TN, FP, FN, TP từ confusion matrix"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)


def create_summary_statistics(metrics_list: List[Dict]) -> Dict:
    """
    Tính thống kê tổng quát từ danh sách metrics
    (e.g., từ 5-fold CV)
    """
    metrics_array = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1_score': [],
        'auc_roc': [],
    }
    
    for m in metrics_list:
        for key in metrics_array.keys():
            metrics_array[key].append(m[key])
    
    summary = {}
    for key, values in metrics_array.items():
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)
        summary[f'{key}_min'] = np.min(values)
        summary[f'{key}_max'] = np.max(values)
    
    return summary
