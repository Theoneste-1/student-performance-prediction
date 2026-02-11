from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from performance.predictor import predict_performance

def performance_ui(request):
    """
    Render the performance prediction UI with both classifier and regressor results.
    """
    prediction_result = None
    last_data = {}
    error_message = None

    if request.method == 'POST':
        try:
            # Get data from form
            hours_studied = float(request.POST.get('hours_studied', 0))
            previous_scores = float(request.POST.get('previous_scores', 0))
            extracurricular = request.POST.get('extracurricular') == 'on'
            sleep_hours = float(request.POST.get('sleep_hours', 0))
            sample_papers = float(request.POST.get('sample_papers', 0))

            # Store data for form repopulation
            last_data = {
                'hours_studied': hours_studied,
                'previous_scores': previous_scores,
                'extracurricular': extracurricular,
                'sleep_hours': sleep_hours,
                'sample_papers': sample_papers
            }

            # Make unified prediction
            prediction_result = predict_performance(
                hours_studied=hours_studied,
                previous_scores=previous_scores,
                extracurricular=extracurricular,
                sleep_hours=sleep_hours,
                sample_papers=sample_papers
            )

            if not prediction_result['success']:
                error_message = prediction_result['error']

        except ValueError:
            error_message = "Invalid input data. Please enter valid numbers."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render(request, 'performance/index.html', {
        'prediction_result': prediction_result,
        'last_data': last_data,
        'error_message': error_message
    })


@api_view(['POST'])
def predict_performance_api(request):
    """
    API endpoint for performance prediction.
    Returns both student category and performance index.
    
    Request body:
    {
        "hours_studied": float,
        "previous_scores": float,
        "extracurricular": bool,
        "sleep_hours": float,
        "sample_papers": float
    }
    
    Response:
    {
        "success": bool,
        "student_category": str,
        "performance_index": float,
        "schedule_valid": bool,
        "confidence": float,
        "error": str or null,
        "details": {
            "total_daily_hours": float,
            "schedule_analysis": str,
            "input_summary": {...}
        }
    }
    """
    try:
        data = request.data

        # Extract features
        hours_studied = float(data.get('hours_studied', 0))
        previous_scores = float(data.get('previous_scores', 0))
        extracurricular = bool(data.get('extracurricular', False))
        sleep_hours = float(data.get('sleep_hours', 0))
        sample_papers = float(data.get('sample_papers', 0))

        # Get prediction
        result = predict_performance(
            hours_studied=hours_studied,
            previous_scores=previous_scores,
            extracurricular=extracurricular,
            sleep_hours=sleep_hours,
            sample_papers=sample_papers
        )

        return Response(result)

    except (ValueError, TypeError) as e:
        return Response({
            'success': False,
            'student_category': None,
            'performance_index': None,
            'schedule_valid': False,
            'confidence': 0.0,
            'error': f"Invalid input: {str(e)}",
            'details': {}
        }, status=400)
    except Exception as e:
        return Response({
            'success': False,
            'student_category': None,
            'performance_index': None,
            'schedule_valid': False,
            'confidence': 0.0,
            'error': f"Server error: {str(e)}",
            'details': {}
        }, status=500)

