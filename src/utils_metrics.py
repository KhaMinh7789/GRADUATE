"""Utility functions for computing and storing evaluation metrics.
Revised v2: Added AUC-PR, Youden's J statistic, and threshold sweep analysis.
These additions address advisor feedback on metric selection and threshold justification.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
from typing import Dict, Tuple, List
import json
import matplotlib.pyplot as plt


class MetricsCalculator:
    """Compute comprehensive metrics for binary classification."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             y_proba: np.ndarray) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels [0, 1]
            y_pred: Binary predictions [0, 1]
            y_proba: Probability predictions (0-1)
        
        Returns:
            Dict containing all metrics
        """
        # Flatten to ensure correct shape
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
        
        # AUC-ROC (secondary metric for imbalanced data)
        try:
            auc_roc_score = roc_auc_score(y_true, y_proba)
        except Exception:
            auc_roc_score = 0.0
        
        # AUC-PR (PRIMARY metric for imbalanced medical data)
        # AUC-PR focuses on minority class (COVID+) performance
        # More informative than AUC-ROC when prevalence << 0.5
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            auc_pr_score = auc(recall_curve, precision_curve)
        except Exception:
            auc_pr_score = 0.0
        
        # Average Precision Score (area under PR curve, interpolated)
        try:
            avg_precision = average_precision_score(y_true, y_proba)
        except Exception:
            avg_precision = 0.0

        metrics = {
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1_score': f1,
            'auc_roc': auc_roc_score,
            'auc_pr': auc_pr_score,           # PRIMARY metric (new)
            'avg_precision': avg_precision,    # Alternative PR summary (new)
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'ece': MetricsCalculator.calculate_ece(y_true, y_proba)
        }
        
        return metrics
    
    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return float(ece)
    
    @staticmethod
    def create_metrics_table(results_dict: Dict) -> pd.DataFrame:
        """
        Create metrics table from results dictionary.
        
        Args:
            results_dict: Dict with format {dataset_name: metrics}
        
        Returns:
            DataFrame with all metrics
        """
        table_data = []
        
        for dataset_name, metrics in results_dict.items():
            row = {
                'Dataset': dataset_name,
                'AUC-PR': f"{metrics.get('auc_pr', 0):.4f}",    # PRIMARY - first column
                'AUC-ROC': f"{metrics.get('auc_roc', 0):.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Accuracy (ref)': f"{metrics['accuracy']:.4f}",  # kept for reference only
                'ECE': f"{metrics.get('ece', 0):.4f}",
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
        """Save metrics as JSON."""
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics to: {output_path}")
    
    @staticmethod
    def format_metrics_for_paper(metrics: Dict) -> str:
        """Format metrics as single-line string for paper reporting."""
        return (
            f"AUC-PR: {metrics.get('auc_pr', 0):.4f} | "
            f"AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"Sensitivity: {metrics['sensitivity']:.2%} | "
            f"Specificity: {metrics['specificity']:.2%}"
        )


class ThresholdOptimizer:
    """Threshold selection using Youden's J statistic and sweep analysis.
    
    Addresses advisor feedback: threshold tau=0.35 was arbitrarily chosen.
    This class provides scientifically justified selection methodology.
    
    Reference: Youden WJ (1950). 'Index for rating diagnostic tests.'
               Cancer, 3(1):32-35.
    
    Youden's J: J = Sensitivity + Specificity - 1 = TPR - FPR
    The threshold maximizing J gives the best Sensitivity-Specificity balance.
    """
    
    @staticmethod
    def find_optimal_youden(y_true: np.ndarray, 
                            y_proba: np.ndarray) -> Dict:
        """
        Find optimal decision threshold using Youden's J statistic.
        
        Args:
            y_true:  Ground truth labels [0, 1]
            y_proba: Predicted probabilities [0, 1]
            
        Returns:
            Dict with optimal threshold, sensitivity, specificity, Youden J
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_j = tpr - fpr  # Equivalent: Sensitivity + Specificity - 1
        optimal_idx = np.argmax(youden_j)
        
        return {
            'optimal_threshold': float(thresholds[optimal_idx]),
            'sensitivity': float(tpr[optimal_idx]),
            'specificity': float(1 - fpr[optimal_idx]),
            'youden_j': float(youden_j[optimal_idx]),
            'fpr': float(fpr[optimal_idx]),
            'tpr': float(tpr[optimal_idx]),
        }
    
    @staticmethod
    def threshold_sweep(y_true: np.ndarray,
                        y_proba: np.ndarray,
                        thresholds: List[float] = None) -> List[Dict]:
        """
        Evaluate performance at multiple decision thresholds.
        
        Clinical guidance:
        - Low tau (0.20-0.30): High sensitivity -> triage/screening
        - Youden-optimal (0.30-0.45): Balanced -> general use
        - High tau (0.50-0.70): High precision -> confirmation
        
        Args:
            y_true:     Ground truth labels
            y_proba:    Predicted probabilities
            thresholds: List of tau values [default: 0.10..0.70]
            
        Returns:
            List of dicts with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                          0.55, 0.60, 0.65, 0.70]
        
        results = []
        for tau in thresholds:
            y_pred = (np.array(y_proba) >= tau).astype(int)
            y_true_arr = np.array(y_true)
            
            cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
            J = sens + spec - 1  # Youden's J statistic
            
            results.append({
                'threshold': tau,
                'sensitivity': round(sens, 4),
                'specificity': round(spec, 4),
                'precision': round(prec, 4),
                'f1_score': round(f1, 4),
                'youden_j': round(J, 4),
                'tp': int(tp), 'fp': int(fp),
                'tn': int(tn), 'fn': int(fn),
            })
        
        return results
    
    @staticmethod
    def print_sweep_table(sweep_results: List[Dict]):
        """Print formatted threshold sweep table for thesis Table 3."""
        max_j = max(r['youden_j'] for r in sweep_results)
        print(f"{'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} "
              f"{'Precision':>10} {'F1':>8} {'Youden_J':>10}")
        print("-" * 72)
        for r in sweep_results:
            marker = " <- Youden-opt" if r['youden_j'] == max_j else ""
            print(f"  tau={r['threshold']:.2f}  {r['sensitivity']:>10.1%}  "
                  f"{r['specificity']:>10.1%}  {r['precision']:>9.1%}  "
                  f"{r['f1_score']:>6.4f}  {r['youden_j']:>8.4f}{marker}")


class RocCurveCalculator:
    """Compute ROC and PR curves."""
    
    @staticmethod
    def calculate_roc(y_true: np.ndarray, 
                     y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute FPR, TPR, AUC for ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score
    
    @staticmethod
    def calculate_pr_curve(y_true: np.ndarray,
                           y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute Precision, Recall, AUC-PR for PR curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = auc(recall, precision)
        return precision, recall, auc_pr
    
    @staticmethod
    def get_roc_for_multiple_models(results_dict: Dict) -> Dict:
        """Compute ROC curves for multiple models."""
        roc_dict = {}
        for model_name, datasets in results_dict.items():
            roc_dict[model_name] = {}
            for dataset_name, data in datasets.items():
                fpr, tpr, auc_score = RocCurveCalculator.calculate_roc(
                    data['y_true'], data['y_proba']
                )
                roc_dict[model_name][dataset_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc_score
                }
        return roc_dict


class ConfusionMatrixCalculator:
    """Helper for confusion matrix computations."""
    
    @staticmethod
    def get_cm_values(y_true: np.ndarray, 
                     y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Return TN, FP, FN, TP from confusion matrix."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)


def create_summary_statistics(metrics_list: List[Dict]) -> Dict:
    """
    Compute aggregate statistics from a list of fold metrics.
    (e.g., from 5-fold CV). Now includes AUC-PR tracking.
    """
    keys_to_track = ['accuracy', 'sensitivity', 'specificity',
                     'precision', 'f1_score', 'auc_roc', 'auc_pr', 'ece']
    metrics_array = {k: [] for k in keys_to_track}
    
    for m in metrics_list:
        for key in keys_to_track:
            if key in m:
                metrics_array[key].append(m[key])
    
    summary = {}
    for key, values in metrics_array.items():
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))
    
    return summary
