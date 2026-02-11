"""
Train a classifier to categorize students into different study behavior categories.
This classifier predicts: Stressed, Diligent, Relaxed, Balanced, or Idle
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from performance.feature_engineering import add_features

def train_classifier():
    """
    Train a Random Forest classifier to categorize students based on study habits.
    """
    # Load dataset
    df = pd.read_csv('dataset.csv')
    
    # 1. Feature Engineering - Add Student Categories and Schedule Validation
    df = add_features(df)
    
    # 2. Encode Extracurricular Activities
    le_extra = LabelEncoder()
    df['Extracurricular Activities'] = le_extra.fit_transform(df['Extracurricular Activities'])
    
    # 3. Filter out impossible schedules (optional - or keep them for training)
    # We'll keep them to let the model learn about impossible schedules
    print(f"Total records: {len(df)}")
    print(f"Valid schedules: {df['Schedule Valid'].sum()}")
    print(f"Invalid schedules: {(~df['Schedule Valid']).sum()}")
    
    # 4. Prepare features and target
    feature_columns = [
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]
    
    X = df[feature_columns]
    y = df['Student Category']  # Target: Student Category
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train classifier
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)
    
    # 7. Evaluate on test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("CLASSIFIER PERFORMANCE")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 8. Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(feature_columns, classifier.feature_importances_):
        print(f"  {feature}: {importance:.4f}")
    
    # 9. Save the classifier and encoders
    joblib.dump(classifier, 'performance/classifier.pkl')
    joblib.dump(le_extra, 'performance/extracurricular_encoder.pkl')
    
    print("\n" + "="*60)
    print("✓ Classifier trained and saved successfully!")
    print("  - Model: performance/classifier.pkl")
    print("  - Encoder: performance/extracurricular_encoder.pkl")
    print("="*60)
    
    return classifier, le_extra

if __name__ == '__main__':
    train_classifier()
