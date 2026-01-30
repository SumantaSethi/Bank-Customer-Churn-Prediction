Flask Application (app.py)
Main web application for Bank Churn Prediction

IMPORT LIBRARIES

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os

INITIALIZE FLASK APP

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

LOAD ALL MODELS

print("Loading models...")

try:
    # Load traditional ML models
    model_logistic = joblib.load('models/logistic_regression.pkl')
    model_rf = joblib.load('models/random_forest.pkl')
    model_svm = joblib.load('models/svm_smote.pkl')
    model_xgb = joblib.load('models/xgboost.pkl')
    
    # Load Deep Learning model
    model_dl = keras.models.load_model('models/deep_learning.h5')
    
    # Load encoders
    le_geography = joblib.load('models/label_encoder_geography.pkl')
    le_gender = joblib.load('models/label_encoder_gender.pkl')
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")

HELPER FUNCTIONS

def preprocess_input(data):
    """
    Preprocess user input data for model prediction
    
    Args:
        data: Dictionary containing user input
        
    Returns:
        numpy array ready for prediction
    """
    # Encode Geography
    geography_mapping = {
        'France': 0,
        'Germany': 1,
        'Spain': 2
    }
    
    # Encode Gender
    gender_mapping = {
        'Male': 0,
        'Female': 1
    }
    
    # Create feature array in correct order
    features = np.array([[
        float(data['credit_score']),
        geography_mapping[data['geography']],
        gender_mapping[data['gender']],
        float(data['age']),
        float(data['tenure']),
        float(data['balance']),
        int(data['num_products']),
        int(data['has_credit_card']),
        int(data['is_active_member']),
        float(data['estimated_salary'])
    ]])
    
    return features

def get_risk_level(probability):
    """
    Determine risk level based on churn probability
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        tuple: (risk_level, color, message)
    """
    if probability < 0.3:
        return 'Low Risk', 'success', 'Customer is likely to stay'
    elif probability < 0.6:
        return 'Medium Risk', 'warning', 'Customer may need attention'
    else:
        return 'High Risk', 'danger', 'Immediate retention action recommended'


ROUTES


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page and endpoint"""
    
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'credit_score': request.form.get('credit_score'),
                'geography': request.form.get('geography'),
                'gender': request.form.get('gender'),
                'age': request.form.get('age'),
                'tenure': request.form.get('tenure'),
                'balance': request.form.get('balance'),
                'num_products': request.form.get('num_products'),
                'has_credit_card': request.form.get('has_credit_card'),
                'is_active_member': request.form.get('is_active_member'),
                'estimated_salary': request.form.get('estimated_salary')
            }
            
            # Validate all fields are present
            if any(value is None for value in data.values()):
                return render_template('predict.html', 
                                     error="Please fill in all fields")
            
            # Preprocess input
            features = preprocess_input(data)
            
            # Get predictions from all models
            predictions = {}
            
            # Logistic Regression
            try:
                prob_lr = model_logistic.predict_proba(features)[0][1]
                predictions['Logistic Regression'] = {
                    'probability': round(prob_lr * 100, 2),
                    'prediction': 'Will Churn' if prob_lr > 0.5 else 'Will Stay'
                }
            except Exception as e:
                predictions['Logistic Regression'] = {'error': str(e)}
            
            # Random Forest
            try:
                prob_rf = model_rf.predict_proba(features)[0][1]
                predictions['Random Forest'] = {
                    'probability': round(prob_rf * 100, 2),
                    'prediction': 'Will Churn' if prob_rf > 0.5 else 'Will Stay'
                }
            except Exception as e:
                predictions['Random Forest'] = {'error': str(e)}
            
            # SVM
            try:
                prob_svm = model_svm.predict_proba(features)[0][1]
                predictions['SVM (SMOTE)'] = {
                    'probability': round(prob_svm * 100, 2),
                    'prediction': 'Will Churn' if prob_svm > 0.5 else 'Will Stay'
                }
            except Exception as e:
                predictions['SVM (SMOTE)'] = {'error': str(e)}
            
            # XGBoost
            try:
                prob_xgb = model_xgb.predict_proba(features)[0][1]
                predictions['XGBoost'] = {
                    'probability': round(prob_xgb * 100, 2),
                    'prediction': 'Will Churn' if prob_xgb > 0.5 else 'Will Stay'
                }
            except Exception as e:
                predictions['XGBoost'] = {'error': str(e)}
            
            # Deep Learning
            try:
                prob_dl = model_dl.predict(features, verbose=0)[0][0]
                predictions['Deep Learning'] = {
                    'probability': round(float(prob_dl) * 100, 2),
                    'prediction': 'Will Churn' if prob_dl > 0.5 else 'Will Stay'
                }
            except Exception as e:
                predictions['Deep Learning'] = {'error': str(e)}
            
            # Calculate average probability
            valid_probs = [
                p['probability'] for p in predictions.values() 
                if 'probability' in p
            ]
            
            if valid_probs:
                avg_probability = round(np.mean(valid_probs), 2)
                risk_level, risk_color, risk_message = get_risk_level(avg_probability / 100)
            else:
                avg_probability = None
                risk_level = risk_color = risk_message = None
            
            # Prepare customer data for display
            customer_data = {
                'credit_score': data['credit_score'],
                'geography': data['geography'],
                'gender': data['gender'],
                'age': data['age'],
                'tenure': data['tenure'],
                'balance': f"${float(data['balance']):,.2f}",
                'num_products': data['num_products'],
                'has_credit_card': 'Yes' if int(data['has_credit_card']) == 1 else 'No',
                'is_active_member': 'Yes' if int(data['is_active_member']) == 1 else 'No',
                'estimated_salary': f"${float(data['estimated_salary']):,.2f}"
            }
            
            return render_template('results.html',
                                 predictions=predictions,
                                 avg_probability=avg_probability,
                                 risk_level=risk_level,
                                 risk_color=risk_color,
                                 risk_message=risk_message,
                                 customer_data=customer_data)
            
        except Exception as e:
            return render_template('predict.html', 
                                 error=f"An error occurred: {str(e)}")
    
    # GET request - show the form
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for programmatic predictions
    
    Expected JSON format:
    {
        "credit_score": 600,
        "geography": "France",
        "gender": "Male",
        "age": 40,
        "tenure": 5,
        "balance": 100000,
        "num_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 50000
    }
    """
    try:
        data = request.get_json()
        
        # Preprocess
        features = preprocess_input(data)
        
        # Get predictions
        results = {
            'logistic_regression': float(model_logistic.predict_proba(features)[0][1]),
            'random_forest': float(model_rf.predict_proba(features)[0][1]),
            'svm': float(model_svm.predict_proba(features)[0][1]),
            'xgboost': float(model_xgb.predict_proba(features)[0][1]),
            'deep_learning': float(model_dl.predict(features, verbose=0)[0][0])
        }
        
        # Average probability
        avg_prob = np.mean(list(results.values()))
        
        # Risk assessment
        risk_level, _, risk_message = get_risk_level(avg_prob)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'average_probability': float(avg_prob),
            'risk_level': risk_level,
            'risk_message': risk_message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/models-info')
def models_info():
    """Model information page"""
    return render_template('models_info.html')


ERROR HANDLERS

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


# RUN APP


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
