from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load all models
print("Loading models...")

try:
    model_logistic = joblib.load('models/logistic_regression.pkl')
    model_rf = joblib.load('models/random_forest.pkl')
    model_svm = joblib.load('models/svm_smote.pkl')
    model_xgb = joblib.load('models/xgboost.pkl')
    model_dl = keras.models.load_model('models/deep_learning.h5')
    le_geography = joblib.load('models/label_encoder_geography.pkl')
    le_gender = joblib.load('models/label_encoder_gender.pkl')
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")


def preprocess_input(data):
    """Preprocess user input data for model prediction"""
    geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_mapping = {'Male': 0, 'Female': 1}
    
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
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return 'Low Risk', 'success', 'Customer is likely to stay'
    elif probability < 0.6:
        return 'Medium Risk', 'warning', 'Customer may need attention'
    else:
        return 'High Risk', 'danger', 'Immediate retention action recommended'

def _get_field(form, name, type_func=str, required=True, min_val=None, max_val=None, allowed=None):
    """
    Safely extract and validate a field from request.form.

    - form: request.form
    - name: field name
    - type_func: conversion function (int, float, str)
    - required: if True, raise ValueError on missing/empty value
    - min_val / max_val: numeric range checks (applies after conversion)
    - allowed: set or list of allowed values (applies after conversion)
    """
    val = form.get(name)
    if required and (val is None or str(val).strip() == ""):
        raise ValueError(f"Missing required field: {name}")
    if val is None or str(val).strip() == "":
        return None
    try:
        parsed = type_func(val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid value for {name}: {val!r}")
    if min_val is not None and parsed < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and parsed > max_val:
        raise ValueError(f"{name} must be <= {max_val}")
    if allowed is not None and parsed not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return parsed


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
            credit_score = request.form.get('credit_score')
            geography = request.form.get('geography')
            gender = request.form.get('gender')
            age = request.form.get('age')
            tenure = request.form.get('tenure')
            balance = request.form.get('balance')
            num_products = request.form.get('num_products')
            has_credit_card = request.form.get('has_credit_card')
            is_active_member = request.form.get('is_active_member')
            estimated_salary = request.form.get('estimated_salary')
            
            # Check all fields are present
            if not all([credit_score, geography, gender, age, tenure, balance, 
                       num_products, has_credit_card, is_active_member, estimated_salary]):
                return render_template('predict.html', 
                                     error="Please fill in all fields")
            
            # Convert to proper types
            try:
                credit_score = int(credit_score)
                age = int(age)
                tenure = int(tenure)
                balance = float(balance)
                num_products = int(num_products)
                has_credit_card = int(has_credit_card)
                is_active_member = int(is_active_member)
                estimated_salary = float(estimated_salary)
            except ValueError:
                return render_template('predict.html', 
                                     error="Invalid number format in one or more fields")
            
            # Validate ranges
            if not (300 <= credit_score <= 850):
                return render_template('predict.html', 
                                     error="Credit score must be between 300 and 850")
            
            if not (18 <= age <= 100):
                return render_template('predict.html', 
                                     error="Age must be between 18 and 100")
            
            if not (0 <= tenure <= 10):
                return render_template('predict.html', 
                                     error="Tenure must be between 0 and 10 years")
            
            # Prepare data
            data = {
                'credit_score': credit_score,
                'geography': geography,
                'gender': gender,
                'age': age,
                'tenure': tenure,
                'balance': balance,
                'num_products': num_products,
                'has_credit_card': has_credit_card,
                'is_active_member': is_active_member,
                'estimated_salary': estimated_salary
            }
            
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
            valid_probs = [p['probability'] for p in predictions.values() if 'probability' in p]
            
            if valid_probs:
                avg_probability = round(np.mean(valid_probs), 2)
                risk_level, risk_color, risk_message = get_risk_level(avg_probability / 100)
            else:
                avg_probability = None
                risk_level = risk_color = risk_message = None
            
            # Prepare customer data for display
            customer_data = {
                'credit_score': credit_score,
                'geography': geography,
                'gender': gender,
                'age': age,
                'tenure': tenure,
                'balance': f"${float(balance):,.2f}",
                'num_products': num_products,
                'has_credit_card': 'Yes' if has_credit_card == 1 else 'No',
                'is_active_member': 'Yes' if is_active_member == 1 else 'No',
                'estimated_salary': f"${float(estimated_salary):,.2f}"
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
    
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        data = request.get_json()
        features = preprocess_input(data)
        
        results = {
            'logistic_regression': float(model_logistic.predict_proba(features)[0][1]),
            'random_forest': float(model_rf.predict_proba(features)[0][1]),
            'svm': float(model_svm.predict_proba(features)[0][1]),
            'xgboost': float(model_xgb.predict_proba(features)[0][1]),
            'deep_learning': float(model_dl.predict(features, verbose=0)[0][0])
        }
        
        avg_prob = np.mean(list(results.values()))
        risk_level, _, risk_message = get_risk_level(avg_prob)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'average_probability': float(avg_prob),
            'risk_level': risk_level,
            'risk_message': risk_message
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/about')
def about():
    """About page"""
    try:
        return render_template('about.html')
    except:
        return "<h1>About Page</h1><p>Bank Churn Prediction System</p>"


@app.route('/models-info')
def models_info():
    """Model information page"""
    try:
        return render_template('models_info.html')
    except:
        return "<h1>Models Info</h1><p>Using 6 ML models for prediction</p>"


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404


@app.errorhandler(500)
def internal_error(e):
    return "<h1>500 - Internal Server Error</h1>", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
