from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow import keras
import os
import logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configure logger
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Global variables for models and encoders
model_logistic = None
model_rf = None
model_svm = None
model_xgb = None
model_dl = None
le_geography = None
le_gender = None

# Load models at startup - fail loudly if critical pieces are missing
print("Loading models...")
models_loaded = True

try:
    model_logistic = joblib.load('models/logistic_regression.pkl')
    model_rf = joblib.load('models/random_forest.pkl')
    model_svm = joblib.load('models/svm_smote.pkl')
    model_xgb = joblib.load('models/xgboost.pkl')
    model_dl = keras.models.load_model('models/deep_learning.h5')
    le_geography = joblib.load('models/label_encoder_geography.pkl')
    le_gender = joblib.load('models/label_encoder_gender.pkl')
    print("✅ All models and encoders loaded successfully!")
except Exception as e:
    models_loaded = False
    error_msg = f"❌ Critical error loading models/encoders: {str(e)}"
    print(error_msg)
    app.logger.exception("Model loading failed")

if not models_loaded:
    @app.route('/')
    @app.route('/predict', methods=['GET', 'POST'])
    @app.route('/api/predict', methods=['POST'])
    def show_error():
        return f"<h1>Server Configuration Error</h1><p>{error_msg}</p><p>Please check model files and restart the application.</p>", 500


def preprocess_input(data):
    """Preprocess user input data for model prediction"""
    if le_geography is None or le_gender is None:
        raise RuntimeError("Label encoders not loaded - cannot process categorical features")

    features = np.array([[
        float(data['credit_score']),
        le_geography.transform([data['geography']])[0],
        le_gender.transform([data['gender']])[0],
        float(data['age']),
        float(data['tenure']),
        float(data['balance']),
        int(data['num_products']),
        int(data['has_credit_card']),
        int(data['is_active_member']),
        float(data['estimated_salary'])
    ]])

    return features


def decode(pred):
    """Decode binary prediction: 1 → Customer Exits, 0 → Customer Stays"""
    return 'Customer Exits' if pred == 1 else 'Customer Stays'


def _get_field(form, name, type_func=str, required=True, min_val=None, max_val=None, allowed=None):
    val = form.get(name)

    if val is None or str(val).strip() == "":
        if required:
            raise ValueError(f"Missing required field: {name}")
        return None

    try:
        parsed = type_func(val)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for {name}: '{val}' ({type(val).__name__}) → {e}")

    if parsed is None:
        raise ValueError(f"Field '{name}' could not be converted (got None after parsing)")

    if min_val is not None and parsed < min_val:
        raise ValueError(f"{name} must be >= {min_val} (got {parsed})")
    if max_val is not None and parsed > max_val:
        raise ValueError(f"{name} must be <= {max_val} (got {parsed})")

    if allowed is not None and parsed not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)} (got {parsed!r})")

    return parsed


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        app.logger.debug("Predict form data: %s", request.form.to_dict())

        try:
            credit_score     = _get_field(request.form, "credit_score",     int,   True, 300, 850)
            age              = _get_field(request.form, "age",              int,   True, 18,  100)
            geography        = _get_field(request.form, "geography",        str,   True, allowed={"France", "Germany", "Spain"})
            gender           = _get_field(request.form, "gender",           str,   True, allowed={"Male", "Female"})
            tenure           = _get_field(request.form, "tenure",           int,   True, 0,   10)
            balance          = _get_field(request.form, "balance",          float, True, 0)
            num_products     = _get_field(request.form, "num_products",     int,   True, 1,   100)
            estimated_salary = _get_field(request.form, "estimated_salary", float, True, 0)
            has_credit_card  = _get_field(request.form, "has_credit_card",  int,   True, 0,   1)
            is_active_member = _get_field(request.form, "is_active_member", int,   True, 0,   1)

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

            features = preprocess_input(data)

            predictions = {}

            # Logistic Regression
            try:
                prob = model_logistic.predict_proba(features)[0][1]
                predictions['Logistic Regression'] = {
                    'probability': round(prob * 100, 2),
                    'prediction': decode(1 if prob > 0.5 else 0)
                }
            except Exception as e:
                app.logger.exception("Logistic failed")
                predictions['Logistic Regression'] = {'error': str(e)}

            # Random Forest
            try:
                prob = model_rf.predict_proba(features)[0][1]
                predictions['Random Forest'] = {
                    'probability': round(prob * 100, 2),
                    'prediction': decode(1 if prob > 0.5 else 0)
                }
            except Exception as e:
                app.logger.exception("RF failed")
                predictions['Random Forest'] = {'error': str(e)}

            # SVM
            try:
                prob = model_svm.predict_proba(features)[0][1]
                predictions['SVM (SMOTE)'] = {
                    'probability': round(prob * 100, 2),
                    'prediction': decode(1 if prob > 0.5 else 0)
                }
            except Exception as e:
                app.logger.exception("SVM failed")
                predictions['SVM (SMOTE)'] = {'error': str(e)}

            # XGBoost
            try:
                prob = model_xgb.predict_proba(features)[0][1]
                predictions['XGBoost'] = {
                    'probability': round(prob * 100, 2),
                    'prediction': decode(1 if prob > 0.5 else 0)
                }
            except Exception as e:
                app.logger.exception("XGB failed")
                predictions['XGBoost'] = {'error': str(e)}

            # Deep Learning
            try:
                prob = model_dl.predict(features, verbose=0)[0][0]
                predictions['Deep Learning'] = {
                    'probability': round(float(prob) * 100, 2),
                    'prediction': decode(1 if prob > 0.5 else 0)
                }
            except Exception as e:
                app.logger.exception("DL failed")
                predictions['Deep Learning'] = {'error': str(e)}

            valid_probs = [p['probability'] for p in predictions.values() if 'probability' in p]
            avg_probability = round(np.mean(valid_probs), 2) if valid_probs else None

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

            app.logger.info("Prediction successful - rendering results.html")
            print("DEBUG: Rendering results.html with avg prob =", avg_probability)

            return render_template('results.html',
                                   predictions=predictions,
                                   avg_probability=avg_probability,
                                   customer_data=customer_data)

        except ValueError as ve:
            app.logger.warning("Validation error: %s", ve)
            return render_template('predict.html', error=str(ve))
        except Exception as e:
            app.logger.exception("Unexpected prediction error")
            return render_template('predict.html', error=f"An error occurred: {str(e)}")

    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
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

        return jsonify({
            'success': True,
            'predictions': results,
            'average_probability': float(avg_prob)
        })

    except Exception as e:
        app.logger.exception("API predict failed")
        return jsonify({'success': False, 'error': str(e)}), 500


# Optional extra pages (can be removed if you want only 2 routes)
@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except:
        return "<h1>About</h1><p>Bank Churn Prediction App</p>"


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
