#!/usr/bin/env python
"""
🎓 Quick Start Script for Student Performance AI System

This script helps you get started with the system in 4 easy steps:
1. Train models
2. Run tests
3. Validate dataset
4. Start web server

Usage:
    python quickstart.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(number, text):
    """Print a formatted step."""
    print(f"\n[{number}/4] {text}")
    print("-" * 70)

def check_django():
    """Check if Django is installed and project exists."""
    try:
        import django
        return True
    except ImportError:
        print("❌ Django not found. Install with: pip install django djangorestframework")
        return False

def check_models():
    """Check if trained models exist."""
    models_dir = Path('performance')
    required_files = [
        'classifier.pkl',
        'regressor.pkl',
        'extracurricular_encoder.pkl'
    ]
    
    return all((models_dir / f).exists() for f in required_files)

def train_models():
    """Train all models."""
    print_step(1, "Training Models")
    
    try:
        from performance.train_all import train_all_models
        success = train_all_models()
        return success
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def run_tests():
    """Run system tests."""
    print_step(2, "Running Tests")
    
    try:
        from performance.model_utils import test_prediction, inspect_models
        test_prediction(verbose=True)
        print("\n" + "="*70)
        inspect_models()
        return True
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        return False

def validate_data():
    """Validate the dataset."""
    print_step(3, "Validating Dataset")
    
    try:
        from performance.model_utils import validate_dataset
        validate_dataset()
        return True
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False

def start_server():
    """Start the development server."""
    print_step(4, "Starting Development Server")
    
    print("\n✅ All systems ready!")
    print("\n🚀 Starting Django development server...")
    print("   Access the web UI at: http://localhost:8000/performance/")
    print("   API endpoint: http://localhost:8000/api/predict-performance/")
    print("   Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, 'manage.py', 'runserver'])
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")

def main():
    """Main quickstart flow."""
    print("\n" + "="*70)
    print("  🎓 STUDENT PERFORMANCE AI - QUICK START")
    print("="*70)
    
    print("\nThis script will:")
    print("  1. Train the classifier and regressor models")
    print("  2. Run comprehensive tests")
    print("  3. Validate the dataset")
    print("  4. Start the development server")
    
    # Check prerequisites
    if not check_django():
        sys.exit(1)
    
    # Check if models already exist
    if check_models():
        print("\n✅ Models already trained!")
        response = input("\nRetrain models? (y/n): ").lower()
        if response != 'y':
            print("Skipping training...")
            skip_training = True
        else:
            skip_training = False
    else:
        print("\n📦 Models not found. Training is required.")
        skip_training = False
    
    # Step 1: Train
    if not skip_training:
        if not train_models():
            print("\n❌ Failed to train models. Exiting.")
            sys.exit(1)
    
    # Step 2: Test
    if not run_tests():
        response = input("\nTests failed. Continue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 3: Validate
    if not validate_data():
        response = input("\nValidation failed. Continue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 4: Start server
    print("\n" + "="*70)
    response = input("\nStart development server? (y/n): ").lower()
    if response == 'y':
        start_server()
    else:
        print("\n✅ Setup complete! To start the server later, run:")
        print("   python manage.py runserver")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
