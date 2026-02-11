import pandas as pd
import numpy as np

def validate_schedule(hours_studied, sleep_hours):
    """
    Validates if a daily schedule is physically possible.
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if hours_studied < 0 or sleep_hours < 0:
        return False, "Study hours and sleep hours cannot be negative."
    
    if hours_studied + sleep_hours > 24:
        return False, f"Impossible schedule: {hours_studied} study + {sleep_hours} sleep = {hours_studied + sleep_hours} hours exceeds 24 hours in a day."
    
    return True, None
 
def categorize_student(hours_studied, sleep_hours, schedule_valid=True):
    """
    Categorizes a student based on their study and sleep habits.
    
    Categories:
    - Impossible: Schedule violates physical constraints (>24 hours/day)
    - Stressed: High Study (> 6) but Low Sleep (< 6)
    - Diligent: High Study (> 6) and Good Sleep (>= 6)
    - Relaxed: Low Study (<= 6) and High Sleep (> 7)
    - Balanced: Everything else
    - Idle: Zero or near-zero study hours
    """
    
    # Check for impossible schedule
    if not schedule_valid:
        return "Impossible"
    
    # Check for zero/minimal study
    if hours_studied == 0:
        return "Idle"
    
    # High study categories
    if hours_studied > 6:
        if sleep_hours < 6:
            return "Stressed"
        else:
            return "Diligent"
    # Low to moderate study
    else:
        if sleep_hours > 7:
            return "Relaxed"
        else:
            return "Balanced"

def add_features(df):
    """
    Adds feature columns to the dataframe including schedule validation and categorization.
    """
    categories = []
    is_valid_schedule = []
    
    for index, row in df.iterrows():
        is_valid, _ = validate_schedule(row['Hours Studied'], row['Sleep Hours'])
        is_valid_schedule.append(is_valid)
        
        cat = categorize_student(
            row['Hours Studied'], 
            row['Sleep Hours'],
            schedule_valid=is_valid
        )
        categories.append(cat)
    
    df['Schedule Valid'] = is_valid_schedule
    df['Student Category'] = categories
    return df

def get_schedule_validity_feature(hours_studied, sleep_hours):
    """
    Returns a numerical feature representing schedule validity.
    Useful for helping the regressor understand impossible schedules.
    
    Returns:
        float: 1.0 if valid, 0.0 if impossible
    """
    is_valid, _ = validate_schedule(hours_studied, sleep_hours)
    return 1.0 if is_valid else 0.0
