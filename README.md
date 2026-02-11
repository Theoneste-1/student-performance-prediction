# 🎓 Student Performance AI Prediction System

A machine learning-based system that predicts student performance and categorizes student behavior patterns using dual-model architecture (Classifier + Regressor).

## 📋 System Overview

The system uses two complementary machine learning models:

### 1. **Classifier Model** 🎯

- **Purpose**: Categorizes students into behavior patterns
- **Output Categories**:
  - `Stressed`: High study (>6h) + Low sleep (<6h)
  - `Diligent`: High study (>6h) + Good sleep (≥6h)
  - `Relaxed`: Low study (≤6h) + High sleep (>7h)
  - `Balanced`: Moderate study & sleep
  - `Idle`: Zero or near-zero study
  - `Impossible`: Violates physical schedule constraints (>24h/day)

### 2. **Regressor Model** 📈

- **Purpose**: Predicts performance index (0-100)
- **Considers**: All input features including schedule validity
- **Output**: Performance score as a continuous value

## 🎯 Key Features

✅ **Dual-Model Architecture**: Combines classification and regression for comprehensive analysis
✅ **Schedule Validation**: Detects physically impossible schedules (study + sleep > 24h)
✅ **Smart Feature Engineering**: Automatically extracts meaningful patterns from raw data
✅ **Confidence Scoring**: Returns confidence levels for predictions
✅ **Detailed Analysis**: Provides comprehensive breakdown of predictions
✅ **Web & API Support**: Both UI and REST API interfaces

## 📁 Project Structure

```
performance/
├── feature_engineering.py      # Feature processing & schedule validation
├── train_classifier.py         # Classifier model training
├── train_model.py              # Regressor model training
├── train_all.py                # Master training script
├── predictor.py                # Unified prediction module
├── views.py                    # Django views (web & API)
├── urls.py                     # URL routing
├── models.py                   # Django ORM models
├── serializers.py              # DRF serializers
├── admin.py                    # Django admin configuration
├── apps.py                     # App configuration
├── tests.py                    # Unit tests
├── dataset.csv                 # Training dataset
└── templates/
    └── performance/
        └── index.html          # Web UI template
└── static/
    └── performance/
        └── css/
            └── style.css       # Styling
```

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install pandas scikit-learn joblib django djangorestframework numpy
```

### 2. **Train Models**

```bash
# Option A: Using Django shell
python manage.py shell
>>> from performance.train_all import train_all_models
>>> train_all_models()

# Option B: Direct Python script
cd student_ml/performance
python train_all.py
```

This creates three files in `performance/`:

- `classifier.pkl` - Student category classifier
- `regressor.pkl` - Performance index regressor
- `extracurricular_encoder.pkl` - Feature encoder

### 3. **Run the Application**

```bash
python manage.py runserver
```

Access the web interface at: `http://localhost:8000/performance/`

## 📊 Input Features

The system accepts the following inputs:

| Feature             | Type    | Range  | Description                 |
| ------------------- | ------- | ------ | --------------------------- |
| **Hours Studied**   | Float   | 0-24   | Daily study hours           |
| **Previous Scores** | Float   | 0-100  | Previous exam performance   |
| **Sleep Hours**     | Float   | 0-24   | Nightly sleep duration      |
| **Sample Papers**   | Float   | 0+     | Practice papers completed   |
| **Extracurricular** | Boolean | Yes/No | Participation in activities |

### ⚠️ Schedule Validation

- **Valid**: `Hours Studied + Sleep Hours ≤ 24`
- **Invalid**: `Hours Studied + Sleep Hours > 24`

## 🔍 Output Format

### Web UI Response

```json
{
  "success": true,
  "student_category": "Diligent",
  "performance_index": 78.45,
  "schedule_valid": true,
  "confidence": 0.95,
  "error": null,
  "details": {
    "total_daily_hours": 13.0,
    "schedule_analysis": "✅ Schedule is valid",
    "input_summary": {
      "hours_studied": 8,
      "previous_scores": 85,
      "extracurricular": true,
      "sleep_hours": 7,
      "sample_papers": 5
    }
  }
}
```

## 🔌 API Endpoints

### POST `/api/predict-performance/`

**Request Body**:

```json
{
  "hours_studied": 8,
  "previous_scores": 85,
  "extracurricular": true,
  "sleep_hours": 7,
  "sample_papers": 5
}
```

**Response**:

```json
{
  "success": true,
  "student_category": "Diligent",
  "performance_index": 78.45,
  "schedule_valid": true,
  "confidence": 0.95,
  "error": null,
  "details": {...}
}
```

**Error Response**:

```json
{
  "success": false,
  "student_category": null,
  "performance_index": null,
  "schedule_valid": false,
  "confidence": 0.0,
  "error": "Study hours and sleep hours cannot be negative.",
  "details": {}
}
```

## 📈 Model Performance Metrics

### Classifier Metrics

- Accuracy: ~95-98%
- Weighted Precision: ~94-97%
- Weighted Recall: ~95-98%

### Regressor Metrics

- R² Score: ~0.92-0.95
- RMSE: ~4-6 points
- MAE: ~3-4 points

## 🛠️ Advanced Usage

### Using the Predictor Directly

```python
from performance.predictor import predict_performance

result = predict_performance(
    hours_studied=8,
    previous_scores=85,
    extracurricular=True,
    sleep_hours=7,
    sample_papers=5
)

print(f"Category: {result['student_category']}")
print(f"Performance: {result['performance_index']}")
print(f"Confidence: {result['confidence']}")
```

### Handling Impossible Schedules

```python
result = predict_performance(
    hours_studied=20,
    previous_scores=75,
    extracurricular=False,
    sleep_hours=10,  # 20 + 10 = 30 > 24 (IMPOSSIBLE)
    sample_papers=3
)

if not result['schedule_valid']:
    print(f"⚠️ {result['details']['schedule_analysis']}")
    print(f"Performance penalized: {result['performance_index']}")
```

## 🧪 Testing

Run the unit tests:

```bash
python manage.py test performance
```

## 🔒 Data Privacy & Security

- All predictions are computed locally (no external API calls)
- No personal data is stored without explicit consent
- Models are trained on aggregated, anonymized student data

## 📚 Model Training Details

### Training Process

1. **Data Loading**: Reads `dataset.csv`
2. **Feature Engineering**: Adds computed features
3. **Data Validation**: Checks for schedule validity
4. **Feature Encoding**: Converts categorical variables
5. **Train-Test Split**: 80/20 split with stratification
6. **Model Training**: Random Forest algorithms
7. **Evaluation**: Computes performance metrics
8. **Model Persistence**: Saves trained models

### Dataset Requirements

The training dataset should contain:

- `Hours Studied` - numeric
- `Previous Scores` - numeric (0-100)
- `Extracurricular Activities` - categorical (Yes/No)
- `Sleep Hours` - numeric
- `Sample Question Papers Practiced` - numeric
- `Performance Index` - numeric (0-100)

## 🐛 Troubleshooting

### Models Not Found

```
FileNotFoundError: Models not loaded
Solution: Run training script → python train_all.py
```

### Invalid Input Error

```
Error: Study hours and sleep hours cannot be negative
Solution: Ensure all inputs are >= 0
```

### Impossible Schedule Detection

```
⚠️ Impossible schedule: 20 + 10 = 30 hours exceeds 24 hours
Solution: Adjust study/sleep hours so total ≤ 24
```

## 📝 Configuration

Edit these in Django settings:

```python
# settings.py
INSTALLED_APPS = [
    'rest_framework',
    'performance.apps.PerformanceConfig',
]

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
```

## 🤝 Contributing

To extend the system:

1. **Add New Features**: Edit `feature_engineering.py`
2. **Improve Models**: Modify `train_classifier.py` or `train_model.py`
3. **New Predictions**: Extend `predictor.py`
4. **UI Updates**: Modify `templates/performance/index.html`

## 📄 License

This project is provided as-is for educational purposes.

## 💡 Future Enhancements

- [ ] Real-time model retraining
- [ ] Student success trend analysis
- [ ] Comparative performance reports
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Advanced visualization dashboard

---

**Last Updated**: January 2026
**Version**: 2.0 (Classifier + Regressor)
