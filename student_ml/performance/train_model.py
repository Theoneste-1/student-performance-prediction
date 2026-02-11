"""
Train a regressor to predict student performance index.
This model predicts the performance score (0-100) based on study habits and background.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from performance.feature_engineering import add_features, get_schedule_validity_feature

def train_regressor():
    """
    Train a Random Forest regressor to predict student performance index.
    """
    # Load dataset
    df = pd.read_csv('dataset.csv')
    
    # 1. Feature Engineering - Add Student Categories and Schedule Validation
    df = add_features(df)
    
    # 2. Encode Categorical Data
    # Extracurricular Activities
    le_extra = LabelEncoder()
    df['Extracurricular Activities'] = le_extra.fit_transform(df['Extracurricular Activities'])
    
    # Add schedule validity as a feature (helps regressor understand impossible schedules)
    df['Schedule Validity'] = df.apply(
        lambda row: get_schedule_validity_feature(row['Hours Studied'], row['Sleep Hours']),
        axis=1
    )
    
    # 3. Prepare features and target
    feature_columns = [
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced',
        'Schedule Validity'  # NEW: Feature to help model understand impossible schedules
    ]
    
    X = df[feature_columns]
    y = df['Performance Index']
    
    print(f"Total records: {len(df)}")
    print(f"Features: {feature_columns}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Train regressor
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(X_train, y_train)
    
    # 6. Evaluate on test set
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("REGRESSOR PERFORMANCE")
    print("="*60)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 7. Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(feature_columns, regressor.feature_importances_):
        print(f"  {feature}: {importance:.4f}")
    
    # 8. Save artifacts
    joblib.dump(regressor, 'performance/regressor.pkl')
    joblib.dump(le_extra, 'performance/extracurricular_encoder.pkl')
    
    print("\n" + "="*60)
    print("✓ Regressor trained and saved successfully!")
    print("  - Model: performance/regressor.pkl")
    print("  - Encoder: performance/extracurricular_encoder.pkl")
    print("="*60)
    
    return regressor, le_extra

if __name__ == '__main__':
    train_regressor()
