#!/usr/bin/env python3
"""
COVID-19 Cough Detection Web App Launcher
=========================================

Quick launcher script for the Streamlit web application.
Handles environment setup and model loading automatically.

Usage:
    python launch_webapp.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'torch', 'torchaudio', 'librosa',
        'numpy', 'matplotlib', 'Pillow'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r',
                os.path.join(os.path.dirname(__file__), 'requirements.txt')
            ])
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"   pip install -r {os.path.join(os.path.dirname(__file__), 'requirements.txt')}")
            return False

    return True

def check_models():
    """Check if trained models exist"""
    base_dir = Path(__file__).parent.parent  # Covid_Cough_Research directory

    # Check for fold models (actual location)
    wav2vec_models = list(base_dir.glob('best_wav2vec_fold_*.pth'))
    ast_models = list(base_dir.glob('best_ast_fold_*.pth'))

    models_found = {
        'Wav2Vec 2.0 + DNDF': len(wav2vec_models) > 0,
        'AST + DNDF': len(ast_models) > 0
    }

    missing_models = []
    for name, found in models_found.items():
        if not found:
            missing_models.append(name)

    if missing_models:
        print("❌ Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n🔄 Please run the full experiment first:")
        print("   python run_full_experiment.py --mode full")
        return False

    print("✅ All model files found!")
    return True

def launch_app(port=8501, headless=False):
    """Launch the Streamlit app"""
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')

    cmd = [sys.executable, '-m', 'streamlit', 'run', app_path]

    if port != 8501:
        cmd.extend(['--server.port', str(port)])

    if headless:
        cmd.extend(['--server.headless', 'true'])

    print(f"🚀 Launching COVID-19 Cough Detection Web App...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   URL: http://localhost:{port}")
    print("\n" + "="*60)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    parser = argparse.ArgumentParser(description='Launch COVID-19 Cough Detection Web App')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the app on')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--skip-checks', action='store_true', help='Skip requirement and model checks')

    args = parser.parse_args()

    print("🩺 COVID-19 Cough Detection Web App Launcher")
    print("="*60)

    # Check requirements and models
    if not args.skip_checks:
        print("🔍 Checking requirements...")
        if not check_requirements():
            return

        print("\n🔍 Checking models...")
        if not check_models():
            return

    print("\n✅ All checks passed!")
    print("="*60)

    # Launch app
    launch_app(port=args.port, headless=args.headless)

if __name__ == "__main__":
    main()