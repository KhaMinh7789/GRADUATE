"""
RUNNER: Execute all phases (Evaluation, XAI, Report) in sequence
Master script để chốt phần thực nghiệm hoàn toàn
"""
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path


def print_header(phase_name: str, phase_num: int = 0):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {'🚀' if phase_num else '🎯'} {phase_name}")
    print("="*70)


def print_summary():
    """Print experiment summary"""
    summary = """
╔════════════════════════════════════════════════════════════════════╗
║                 COVID-19 COUGH DETECTION                          ║
║              COMPREHENSIVE EXPERIMENT RUNNER                      ║
╚════════════════════════════════════════════════════════════════════╝

📋 Research Objectives:
  ✅ [DONE]  Objective 1: Develop hybrid CNN+DNDF models
  ✅ [DONE]  Objective 2: 5-Fold CV training
  ⏳ [NOW]   Objective 2: Cross-dataset evaluation
  ⏳ [NOW]   Objective 3: XAI Interpretability (LIME)
  ⏳ [NOW]   Generate comprehensive report
  ⏭️  [NEXT]  Objective 4: Web/Mobile App deployment

📊 Execution Plan:
  Phase 1: Cross-Dataset Evaluation        (~5-10 minutes)
  Phase 2: XAI Interpretability (LIME)     (~10-15 minutes)
  Phase 3: Generate Experiment Report      (~5 minutes)
  ────────────────────────────────────────────────────
  TOTAL TIME: ~20-30 minutes

🎯 Outputs:
  ✓ Model comparison metrics table
  ✓ ROC curves comparison
  ✓ Confusion matrix visualizations
  ✓ LIME feature importance visualizations
  ✓ Comprehensive markdown report (EXPERIMENT_REPORT.md)
  ✓ Summary statistics (JSON)

📂 Output Directory: evaluation_results/ → experiment_report/
"""
    print(summary)


def run_phase_1_evaluation():
    """Phase 1: Cross-Dataset Evaluation"""
    print_header("PHASE 1: CROSS-DATASET EVALUATION", 1)
    print("📊 Evaluating Wav2Vec 2.0 + DNDF and AST + DNDF models...")
    print("📂 Loading trained models and datasets...")
    
    try:
        from evaluate_cross_dataset import main as run_evaluation
        run_evaluation()
        print("\n✅ Phase 1 completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Phase 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase_2_xai():
    """Phase 2: XAI Interpretability"""
    print_header("PHASE 2: XAI INTERPRETABILITY (LIME)", 2)
    print("🔍 Generating LIME explanations for model predictions...")
    print("📸 Creating feature importance visualizations...")
    
    try:
        from xai_interpretability import main as run_xai
        run_xai()
        print("\n✅ Phase 2 completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Phase 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase_3_report():
    """Phase 3: Generate Report"""
    print_header("PHASE 3: GENERATE COMPREHENSIVE REPORT", 3)
    print("📝 Compiling evaluation results...")
    print("📊 Creating comparison visualizations...")
    print("📄 Generating markdown report...")
    
    try:
        from generate_experiment_report import main as run_report
        run_report()
        print("\n✅ Phase 3 completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Phase 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_demo():
    """Quick demo (chỉ chạy evaluation, skip XAI)"""
    print_header("QUICK DEMO MODE", 1)
    print("⚡ Running evaluation only (skip XAI for speed)...")
    
    success = run_phase_1_evaluation()
    if success:
        success = run_phase_3_report()
    
    return success


def run_full_pipeline():
    """Run full pipeline"""
    print_summary()
    
    print("\n📋 Press ENTER to continue or CTRL+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Execution cancelled by user")
        return False
    
    start_time = time.time()
    
    # Phase 1
    success_1 = run_phase_1_evaluation()
    if not success_1:
        print("\n⚠️  Phase 1 failed. Skipping subsequent phases.")
        return False
    
    # Phase 2
    success_2 = run_phase_2_xai()
    if not success_2:
        print("\n⚠️  Phase 2 failed. Continuing to report generation...")
    
    # Phase 3
    success_3 = run_phase_3_report()
    
    elapsed = time.time() - start_time
    
    # Summary
    print_header("🎉 EXECUTION COMPLETE", 0)
    print(f"\n⏱️  Total execution time: {elapsed/60:.1f} minutes")
    print("\n✅ All phases completed successfully!" if (success_1 and success_3) 
          else "\n⚠️  Some phases encountered issues")
    
    return success_1 and success_3


def main():
    parser = argparse.ArgumentParser(
        description="COVID-19 Cough Detection - Comprehensive Experiment Runner"
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'demo', 'phase1', 'phase2', 'phase3'],
        default='full',
        help='Execution mode'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                   COVID-19 COUGH DETECTION                        ║
    ║              COMPREHENSIVE EXPERIMENT RUNNER v1.0                 ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Mode: {args.mode.upper()}")
    
    # Execute based on mode
    if args.mode == 'full':
        success = run_full_pipeline()
    
    elif args.mode == 'demo':
        success = run_quick_demo()
    
    elif args.mode == 'phase1':
        print_header("PHASE 1: EVALUATION ONLY", 1)
        success = run_phase_1_evaluation()
    
    elif args.mode == 'phase2':
        print_header("PHASE 2: XAI ONLY", 2)
        success = run_phase_2_xai()
    
    elif args.mode == 'phase3':
        print_header("PHASE 3: REPORT ONLY", 3)
        success = run_phase_3_report()
    
    print(f"\n🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n" + "="*70)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📂 Output files:")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'experiment_report')
        
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                print(f"   • {file}")
        
        print("\n📖 Read the report:")
        report_path = os.path.join(output_dir, 'EXPERIMENT_REPORT.md')
        if os.path.exists(report_path):
            print(f"   👉 {report_path}")
        
        return 0
    else:
        print("\n" + "="*70)
        print("❌ EXPERIMENT FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
