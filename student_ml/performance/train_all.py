"""
Master training script that trains both classifier and regressor models.
Run this script to train all models needed for the student performance predictor.

Usage:
    python manage.py shell
    >>> from performance.train_all import train_all_models
    >>> train_all_models()
    
Or from command line:
    python train_all.py
"""

import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def train_all_models():
    """
    Train all models required for the student performance prediction system.
    """
    print("\n" + "="*70)
    print(" STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING")
    print("="*70)
    
    try:
        # Import training functions
        from performance.train_classifier import train_classifier
        from performance.train_model import train_regressor
        
        print("\n[1/2] Training Student Category Classifier...")
        print("-" * 70)
        classifier, extra_encoder = train_classifier()
        
        print("\n" + "="*70)
        print("\n[2/2] Training Performance Index Regressor...")
        print("-" * 70)
        regressor, extra_encoder = train_regressor()
        
        print("\n" + "="*70)
        print(" ✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        print("\nModel Artifacts Created:")
        print("  1. performance/classifier.pkl          (Student category classifier)")
        print("  2. performance/regressor.pkl           (Performance index regressor)")
        print("  3. performance/extracurricular_encoder.pkl (Feature encoder)")
        print("\nYou can now use the prediction system!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_all_models()
    sys.exit(0 if success else 1)
