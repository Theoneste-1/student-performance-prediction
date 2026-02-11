"""
Utility script for managing and testing the prediction system.
Provides functions for model inspection, validation, and testing.

Usage:
    python manage.py shell
    >>> from performance.model_utils import *
    >>> test_prediction()
    >>> inspect_models()
"""

import numpy as np
import pandas as pd
import joblib
from performance.predictor import StudentPerformancePredictor

def load_models():
    """Load and return all trained models."""
    try:
        classifier = joblib.load('performance/classifier.pkl')
        regressor = joblib.load('performance/regressor.pkl')
        encoder = joblib.load('performance/extracurricular_encoder.pkl')
        return classifier, regressor, encoder
    except FileNotFoundError as e:
        print(f"❌ Models not found: {e}")
        print("Please run: python performance/train_all.py")
        return None, None, None

def inspect_models():
    """Inspect and display model information."""
    print("\n" + "="*60)
    print("MODEL INSPECTION REPORT")
    print("="*60)
    
    classifier, regressor, encoder = load_models()
    
    if classifier is None:
        return
    
    print("\n📊 CLASSIFIER MODEL")
    print("-" * 60)
    print(f"Type: {type(classifier).__name__}")
    print(f"Number of Estimators: {classifier.n_estimators}")
    print(f"Max Depth: {classifier.max_depth}")
    print(f"Classes: {classifier.classes_}")
    print(f"Feature Importance:")
    for idx, (name, importance) in enumerate(zip(
        ['Hours Studied', 'Previous Scores', 'Extracurricular', 
         'Sleep Hours', 'Sample Papers'],
        classifier.feature_importances_
    )):
        print(f"  {idx+1}. {name}: {importance:.4f}")
    
    print("\n📈 REGRESSOR MODEL")
    print("-" * 60)
    print(f"Type: {type(regressor).__name__}")
    print(f"Number of Estimators: {regressor.n_estimators}")
    print(f"Max Depth: {regressor.max_depth}")
    print(f"Feature Importance:")
    for idx, (name, importance) in enumerate(zip(
        ['Hours Studied', 'Previous Scores', 'Extracurricular',
         'Sleep Hours', 'Sample Papers', 'Schedule Validity'],
        regressor.feature_importances_
    )):
        print(f"  {idx+1}. {name}: {importance:.4f}")
    
    print("\n" + "="*60)

def test_prediction(verbose=True):
    """Test the prediction system with sample data."""
    print("\n" + "="*60)
    print("PREDICTION TEST SUITE")
    print("="*60)
    
    predictor = StudentPerformancePredictor()
    
    test_cases = [
        {
            'name': '✅ Valid Diligent Student',
            'hours_studied': 8,
            'previous_scores': 85,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 5
        },
        {
            'name': '⚠️ Stressed Student',
            'hours_studied': 9,
            'previous_scores': 72,
            'extracurricular': False,
            'sleep_hours': 4,
            'sample_papers': 3
        },
        {
            'name': '😴 Relaxed Student',
            'hours_studied': 3,
            'previous_scores': 65,
            'extracurricular': True,
            'sleep_hours': 9,
            'sample_papers': 1
        },
        {
            'name': '🔴 Impossible Schedule',
            'hours_studied': 20,
            'previous_scores': 75,
            'extracurricular': False,
            'sleep_hours': 10,
            'sample_papers': 4
        },
        {
            'name': '⛔ Idle Student',
            'hours_studied': 0,
            'previous_scores': 50,
            'extracurricular': False,
            'sleep_hours': 8,
            'sample_papers': 0
        },
        {
            'name': '❌ Negative Input',
            'hours_studied': -5,
            'previous_scores': 80,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 60)
        
        result = predictor.predict(
            test['hours_studied'],
            test['previous_scores'],
            test['extracurricular'],
            test['sleep_hours'],
            test['sample_papers']
        )
        
        if result['success']:
            print(f"✓ Prediction Successful")
            print(f"  Category: {result['student_category']}")
            print(f"  Performance: {result['performance_index']}/100")
            print(f"  Schedule Valid: {result['schedule_valid']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            if result['details']['schedule_analysis']:
                print(f"  Note: {result['details']['schedule_analysis']}")
        else:
            print(f"✗ Prediction Failed")
            print(f"  Error: {result['error']}")
    
    print("\n" + "="*60)

def validate_dataset(csv_path='dataset.csv'):
    """Validate the training dataset."""
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n📊 Dataset Shape: {df.shape}")
        print(f"   Rows: {df.shape[0]}")
        print(f"   Columns: {df.shape[1]}")
        
        print(f"\n📋 Column Information:")
        print(df.info())
        
        print(f"\n📈 Statistical Summary:")
        print(df.describe())
        
        print(f"\n✅ Missing Values:")
        print(df.isnull().sum())
        
        # Check for impossible schedules
        impossible = (df['Hours Studied'] + df['Sleep Hours'] > 24).sum()
        print(f"\n⚠️ Impossible Schedules Found: {impossible}")
        
        # Check data ranges
        print(f"\n📌 Data Ranges Check:")
        print(f"   Hours Studied: [{df['Hours Studied'].min()}, {df['Hours Studied'].max()}]")
        print(f"   Sleep Hours: [{df['Sleep Hours'].min()}, {df['Sleep Hours'].max()}]")
        print(f"   Previous Scores: [{df['Previous Scores'].min()}, {df['Previous Scores'].max()}]")
        print(f"   Performance Index: [{df['Performance Index'].min()}, {df['Performance Index'].max()}]")
        
        print("\n" + "="*60)
        
    except FileNotFoundError:
        print(f"❌ Dataset not found: {csv_path}")
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")

def compare_predictions(student_data):
    """
    Compare predictions with expected values.
    
    Args:
        student_data (dict): Student information with expected performance
    """
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    
    predictor = StudentPerformancePredictor()
    result = predictor.predict(
        student_data['hours_studied'],
        student_data['previous_scores'],
        student_data['extracurricular'],
        student_data['sleep_hours'],
        student_data['sample_papers']
    )
    
    if result['success']:
        print(f"\nInput Data:")
        print(f"  Study: {student_data['hours_studied']}h")
        print(f"  Previous: {student_data['previous_scores']}/100")
        print(f"  Sleep: {student_data['sleep_hours']}h")
        print(f"  Papers: {student_data['sample_papers']}")
        
        print(f"\nPrediction Results:")
        print(f"  Category: {result['student_category']}")
        print(f"  Performance: {result['performance_index']}/100")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        
        if 'expected_performance' in student_data:
            expected = student_data['expected_performance']
            actual = result['performance_index']
            error = abs(expected - actual)
            print(f"\nComparison:")
            print(f"  Expected: {expected}/100")
            print(f"  Actual: {actual}/100")
            print(f"  Error: {error:.2f} points")
    else:
        print(f"❌ Prediction failed: {result['error']}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    print("\n🎓 Student Performance Prediction System - Utilities\n")
    print("Available functions:")
    print("  - inspect_models()          : Display model information")
    print("  - test_prediction()         : Run test suite")
    print("  - validate_dataset()        : Validate training data")
    print("  - compare_predictions()     : Compare with expected values")
    print("  - load_models()             : Load trained models\n")
    
    # Run tests by default
    test_prediction()
    inspect_models()
    validate_dataset()
