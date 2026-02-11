"""
Unified prediction module that combines classifier and regressor predictions.
Returns both student category and performance index.
"""

import joblib
import numpy as np
from performance.feature_engineering import validate_schedule, get_schedule_validity_feature


class StudentPerformancePredictor:
    """
    Unified predictor that uses both classifier and regressor models.

    Models used:
    - Classifier: Predicts student category (Stressed, Diligent, Relaxed, Balanced, Idle, Impossible)
    - Regressor: Predicts performance index (0-100)
    """

    def __init__(self):
        """Initialize and load pre-trained models."""
        try:
            self.classifier = joblib.load('performance/classifier.pkl')
            self.regressor = joblib.load('performance/regressor.pkl')
            self.extracurricular_encoder = joblib.load(
                'performance/extracurricular_encoder.pkl')
            self.models_loaded = True
        except FileNotFoundError as e:
            print(f"Warning: Could not load models - {e}")
            self.models_loaded = False

    def validate_input(self, hours_studied, previous_scores, sleep_hours, sample_papers):
        """
        Validate input parameters.

        Returns:
            tuple: (is_valid: bool, error_message: str or None)
        """
        errors = []

        if hours_studied < 0:
            errors.append("Study hours cannot be negative.")
        if previous_scores < 0 or previous_scores > 100:
            errors.append("Previous scores must be between 0 and 100.")
        if sleep_hours < 0:
            errors.append("Sleep hours cannot be negative.")
        if sample_papers < 0:
            errors.append("Sample papers practiced cannot be negative.")
        if((hours_studied+ sleep_hours) > 24):
             errors.append("The daily working hours can not exceed 24")
        if(sleep_hours == 0):
            errors.append("You can't survive without sleeping")
        if(hours_studied == 0):
            errors.append("You can't perform without studying")
        if errors:
            return False, ", ".join(errors)

        return True, None

    def prepare_features(self, hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
        """
        Prepare feature array for model prediction.

        Returns:
            np.array: Feature vector ready for prediction
        """
        # Encode extracurricular (assuming 1 = Yes, 0 = No)
        extra_encoded = 1 if extracurricular else 0

        features = np.array([[
            hours_studied,
            previous_scores,
            extra_encoded,
            sleep_hours,
            sample_papers
        ]])

        return features

    def prepare_regressor_features(self, hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
        """
        Prepare feature array for regressor (includes schedule validity).

        Returns:
            np.array: Feature vector ready for regressor prediction
        """
        # Encode extracurricular
        extra_encoded = 1 if extracurricular else 0

        # Add schedule validity feature
        schedule_validity = get_schedule_validity_feature(
            hours_studied, sleep_hours)

        features = np.array([[
            hours_studied,
            previous_scores,
            extra_encoded,
            sleep_hours,
            sample_papers,
            schedule_validity
        ]])

        return features

    def predict(self, hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
        """
        Make unified predictions: student category and performance index.

        Args:
            hours_studied (float): Hours spent studying
            previous_scores (float): Previous exam scores (0-100)
            extracurricular (bool): Involved in extracurricular activities
            sleep_hours (float): Hours of sleep per night
            sample_papers (float): Number of sample papers practiced

        Returns:
            dict: {
                'success': bool,
                'student_category': str or None,
                'performance_index': float or None,
                'schedule_valid': bool,
                'confidence': float,  # Confidence level (0-1)
                'error': str or None,
                'details': {
                    'total_daily_hours': float,
                    'schedule_analysis': str
                }
            }
        """

        # Step 1: Validate inputs
        is_valid, error_msg = self.validate_input(
            hours_studied, previous_scores, sleep_hours, sample_papers
        )

        if not is_valid:
            return {
                'success': False,
                'student_category': None,
                'performance_index': None,
                'schedule_valid': False,
                'confidence': 0.0,
                'error': error_msg,
                'details': {'total_daily_hours': hours_studied + sleep_hours}
            }

        # Step 2: Check schedule validity
        schedule_valid, schedule_error = validate_schedule(
            hours_studied, sleep_hours)
        total_daily_hours = hours_studied + sleep_hours

        schedule_analysis = ""
        if not schedule_valid:
            schedule_analysis = f"⚠️ Impossible schedule detected: {hours_studied}h study + {sleep_hours}h sleep = {total_daily_hours}h (exceeds 24h)"

        # Step 3: Prepare features
        classifier_features = self.prepare_features(
            hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers
        )

        regressor_features = self.prepare_regressor_features(
            hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers
        )

        try:
            # Step 4: Predict student category
            student_category = self.classifier.predict(classifier_features)[0]

            # Step 5: Predict performance index
            performance_pred = self.regressor.predict(regressor_features)[0]

            # Step 6: Apply business logic
            # If schedule is impossible, cap performance
            if not schedule_valid:
                # Penalize impossible schedules
                performance_pred = max(0, performance_pred * 0.5)

            # If no study, performance should be very low
            if hours_studied == 0:
                performance_pred = 0.0
                student_category = "Idle"

            # Cap performance at 100
            performance_index = min(100.0, max(0.0, float(performance_pred)))

            # Step 7: Calculate confidence
            # Confidence is higher for valid schedules and reasonable values
            confidence = 1.0 if schedule_valid else 0.7

            return {
                'success': True,
                'student_category': student_category,
                'performance_index': round(performance_index, 2),
                'schedule_valid': schedule_valid,
                'confidence': round(confidence, 2),
                'error': None,
                'details': {
                    'total_daily_hours': total_daily_hours,
                    'schedule_analysis': schedule_analysis,
                    'input_summary': {
                        'hours_studied': hours_studied,
                        'previous_scores': previous_scores,
                        'extracurricular': extracurricular,
                        'sleep_hours': sleep_hours,
                        'sample_papers': sample_papers
                    }
                }
            }

        except Exception as e:
            return {
                'success': False,
                'student_category': None,
                'performance_index': None,
                'schedule_valid': schedule_valid,
                'confidence': 0.0,
                'error': f"Prediction error: {str(e)}",
                'details': {'total_daily_hours': total_daily_hours}
            }


# Global predictor instance
_predictor = None


def get_predictor():
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = StudentPerformancePredictor()
    return _predictor


def predict_performance(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
    """
    Convenience function to make predictions.

    Returns:
        dict: Prediction result (see StudentPerformancePredictor.predict for details)
    """
    predictor = get_predictor()
    print(predictor.predict(hours_studied, previous_scores,
          extracurricular, sleep_hours, sample_papers))
    return predictor.predict(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers)
