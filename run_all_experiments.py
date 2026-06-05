"""
Master Script: Run all final experiments sequentially
=====================================================
This script will sequentially execute the three final experiments 
required to address the committee's feedback.

Running this in one go will save you time from having to manually start each one.
"""
import subprocess
import sys
import time

def run_script(script_name, description):
    print("\n" + "="*80)
    print(f"🚀 STARTING: {description}")
    print(f"📄 Script: {script_name}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Run the script and stream the output to the console
        result = subprocess.run([sys.executable, script_name], check=True)
        
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print(f"✅ COMPLETED: {description}")
        print(f"⏱️ Time taken: {elapsed/60:.2f} minutes")
        print("="*80 + "\n")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "❌"*40)
        print(f"ERROR: {script_name} failed with exit code {e.returncode}.")
        print("Stopping execution of further experiments.")
        print("❌"*40 + "\n")
        sys.exit(1)

def main():
    print("🌟 INITIALIZING MASTER EXPERIMENT RUNNER 🌟")
    print("This will execute 3 critical experiments sequentially.")
    print("Please do not close this terminal until all 3 are finished.")
    time.sleep(3)
    
    # 1. ECE & Reliability Diagram
    run_script(
        "src/experiment_ece_reliability.py", 
        "Experiment 1 - Uncertainty Estimation (ECE & Reliability Diagrams)"
    )
    
    # 2. Label Noise & Expert Subset
    run_script(
        "src/experiment_expert_labels.py", 
        "Experiment 2 - Label Noise Validation (Expert Subset)"
    )
    
    # 3. Frequency Ablation
    run_script(
        "src/experiment_freq_ablation.py", 
        "Experiment 3 - Objective XAI Validation (Frequency Ablation)"
    )
    
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉")
    print("Please check the 'experiment_report' directory for the generated JSON files and charts.")

if __name__ == "__main__":
    main()
