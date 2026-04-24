#!/usr/bin/env python3
"""
QUICK START SCRIPT
Hướng dẫn nhanh để chạy package
"""

import os
import sys
from pathlib import Path


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║           COVID-19 COUGH DETECTION PACKAGE                        ║
║         Comprehensive Experiment Execution Guide                  ║
╚════════════════════════════════════════════════════════════════════╝

📦 PACKAGE CONTENTS
├── run_full_experiment.py          - Master runner script
├── evaluate_cross_dataset.py        - Phase 1: Evaluation
├── xai_interpretability.py          - Phase 2: XAI (LIME)
├── generate_experiment_report.py   - Phase 3: Report
├── utils_metrics.py                 - Utility functions
└── EXPERIMENT_PACKAGE_README.md    - Detailed documentation

✅ STATUS: Ready to execute
⏱️  ESTIMATED TIME: 25-30 minutes (full pipeline)
🎯 OUTPUT: Complete experiment report with visualizations
"""
    print(banner)


def show_usage():
    usage = """
🚀 QUICK START

Option 1: Full Pipeline (Recommended)
────────────────────────────────────────────────────────────
From PowerShell or Terminal:

E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe run_full_experiment.py --mode full

⏱️  Time: ~25 minutes
📊 Output: Everything


Option 2: Quick Demo (Evaluation + Report)
────────────────────────────────────────────────────────────
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe run_full_experiment.py --mode demo

⏱️  Time: ~10 minutes
📊 Output: Metrics + Report (no XAI)


Option 3: Individual Phases
────────────────────────────────────────────────────────────
# Phase 1 only (evaluation)
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe run_full_experiment.py --mode phase1

# Phase 2 only (XAI)
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe run_full_experiment.py --mode phase2

# Phase 3 only (report)
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe run_full_experiment.py --mode phase3


Option 4: Direct Execution
────────────────────────────────────────────────────────────
# Run Phase 1 directly
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe evaluate_cross_dataset.py

# Run Phase 2 directly
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe xai_interpretability.py

# Run Phase 3 directly
E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe generate_experiment_report.py
"""
    print(usage)


def check_requirements():
    print("\n" + "="*70)
    print("✓ CHECKING REQUIREMENTS")
    print("="*70)
    
    requirements = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    missing = []
    installed = []
    
    for module, name in requirements.items():
        try:
            __import__(module)
            installed.append(f"✅ {name}")
        except ImportError:
            missing.append(f"❌ {name}")
    
    for item in installed:
        print(item)
    
    for item in missing:
        print(item)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} packages. Installing...")
        return False
    else:
        print("\n✅ All packages installed!")
        return True


def show_output_locations():
    print("\n" + "="*70)
    print("📂 OUTPUT LOCATIONS")
    print("="*70)
    
    base_dir = Path(__file__).parent
    locations = """
📊 Evaluation Results:
   └─ evaluation_results/
      ├─ model_comparison_table.csv
      ├─ detailed_metrics.json
      └─ roc_curves.json

🔍 XAI Explanations:
   └─ xai_explanations/
      ├─ sample_*_wav2vec_explanation.png
      ├─ sample_*_ast_explanation.png
      └─ xai_summary.txt

📄 Final Report:
   └─ experiment_report/
      ├─ EXPERIMENT_REPORT.md         👈 OPEN THIS!
      ├─ summary_statistics.json
      ├─ metrics_comparison.png
      ├─ roc_curves.png
      └─ confusion_matrices.png
"""
    print(locations)


def show_next_steps():
    print("\n" + "="*70)
    print("📋 AFTER RUNNING THE PACKAGE")
    print("="*70)
    
    steps = """
1️⃣  OPEN THE REPORT
    Open: experiment_report/EXPERIMENT_REPORT.md
    Format: Markdown (readable in any text editor)
    Content:
    - Executive Summary
    - Model Architecture Details
    - Performance Metrics
    - Analysis & Insights
    - XAI Results
    - Recommendations

2️⃣  REVIEW VISUALIZATIONS
    - metrics_comparison.png      → Model comparison charts
    - roc_curves.png              → ROC curves overlay
    - confusion_matrices.png      → Confusion matrices
    - sample_*_explanation.png    → LIME feature importance

3️⃣  COLLECT DATA FOR THESIS
    - Use metrics from: model_comparison_table.csv
    - Use plots from: experiment_report/*.png
    - Use text from: EXPERIMENT_REPORT.md

4️⃣  NEXT PHASE: WEB/MOBILE APP
    - Deploy models using Flask/FastAPI
    - Create real-time inference interface
    - Implement file upload for cough audio
"""
    print(steps)


def show_troubleshooting():
    print("\n" + "="*70)
    print("🔧 TROUBLESHOOTING")
    print("="*70)
    
    issues = """
Q: "ModuleNotFoundError: No module named 'dataset_offline'"
A: Make sure you're in src/ directory:
   cd Covid_Cough_Research\\src

Q: "CUDA out of memory"
A: Edit script and reduce batch size:
   BATCH_SIZE = 8  # Change from 16

Q: "Models weights not found"
A: Ensure best_*.pth files exist in parent directory:
   D:\\GROWTH\\GRADUTION_MASTER\\Covid_Cough_Research\\best_*.pth

Q: "Script takes too long"
A: Run in demo mode (skips XAI):
   --mode demo

Q: How to run from terminal?
A: Use full path to Python executable:
   E:\\Users\\ADMIN\\miniconda3\\envs\\covid_cough\\python.exe script.py
"""
    print(issues)


def main():
    print_banner()
    show_usage()
    
    try:
        check_requirements()
    except Exception as e:
        print(f"Warning during requirements check: {e}")
    
    show_output_locations()
    show_next_steps()
    show_troubleshooting()
    
    print("\n" + "="*70)
    print("✅ READY TO START")
    print("="*70)
    print("\n👉 Copy one of the commands above and run in PowerShell/Terminal\n")


if __name__ == "__main__":
    main()
