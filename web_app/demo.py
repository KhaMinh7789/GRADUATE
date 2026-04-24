#!/usr/bin/env python3
"""
COVID-19 Cough Detection Web App Demo
=====================================

Demonstrates the web app functionality with sample audio.
Creates a simple test interface to verify the app works correctly.

Usage:
    python demo.py
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from web_app.app import CoughDetectionApp

def create_sample_audio(duration=3.0, frequency=1000, sample_rate=16000, filename="sample_cough.wav"):
    """Create a sample audio file for testing"""
    # Generate a simple tone (simulating cough-like sound)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a more complex sound with harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * frequency * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 2 * frequency * t) +  # 2nd harmonic
        0.2 * np.sin(2 * np.pi * 3 * frequency * t)    # 3rd harmonic
    )

    # Add some noise to make it more realistic
    noise = 0.1 * np.random.normal(0, 1, len(audio))
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Save as WAV
    sf.write(filename, audio, sample_rate)
    print(f"✅ Created sample audio: {filename}")
    return filename

def demo_app():
    """Demonstrate the web app functionality"""
    print("🩺 COVID-19 Cough Detection Web App Demo")
    print("="*60)

    # Initialize app
    app = CoughDetectionApp()

    # Load models
    print("🔄 Loading models...")
    if not app.load_models():
        print("❌ Failed to load models. Please run the full experiment first.")
        return

    print("✅ Models loaded successfully!")
    print()

    # Create sample audio
    print("🎵 Creating sample audio for testing...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sample_file = create_sample_audio(filename=tmp_file.name)

    try:
        # Run prediction
        print("🔍 Analyzing sample audio...")
        results = app.predict(sample_file)

        if results:
            print("\n📋 PREDICTION RESULTS:")
            print("-" * 40)

            for model_name, result in results.items():
                print(f"\n{model_name.upper()} Model:")
                print(f"  Prediction: {result['prediction']}")
                print(".1%")
                print(".1%")
                print(".1%")

            # Consensus
            predictions = [r['prediction'] for r in results.values()]
            covid_count = predictions.count('COVID-19')

            print(f"\n🎯 CONSENSUS: {'COVID-19' if covid_count >= len(results) * 0.5 else 'Healthy'}")
            print(f"   ({covid_count}/{len(results)} models predict COVID-19)")

            print("\n✅ Demo completed successfully!")
            print("\n🚀 To run the full web app:")
            print("   cd web_app")
            print("   python launch_webapp.py")

        else:
            print("❌ Prediction failed")

    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.unlink(sample_file)

if __name__ == "__main__":
    demo_app()