import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load models once at startup
loaded_models = {
    'Decision Tree': joblib.load('models/nate_decision_tree.sav'),
    'K-nearest Neighbors': joblib.load('models/nate_knn.sav'),
    'Logistic Regression': joblib.load('models/nate_logistic_regression.sav'),
    'Random Forest': joblib.load('models/nate_random_forest.sav'),
    'SVM': joblib.load('models/SVM_model.sav'),
    'XGBoost': joblib.load('models/XGBoost_model.sav')
}

def decode(pred):
    return 'Customer Exits' if pred == 1 else 'Customer Stays'

@app.route('/')
def home():
    # Placeholder state for first load
    predictions = [{'model': name, 'prediction': '—'} for name in loaded_models.keys()]
    return render_template('index.html', maind={'customer': {}, 'predictions': predictions})

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Extract and convert values to float (Critical for Scikit-Learn/XGBoost)
    try:
        raw_values = list(request.form.values())
        numeric_values = [float(x) for x in raw_values] 
        new_array = np.array(numeric_values).reshape(1, -1)
    except ValueError as e:
        return f"Error: Please ensure all inputs are numeric. {e}", 400

    # 2. Build Customer Display Dictionary
    cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    custd = dict(zip(cols, raw_values))
    
    # Prettify Yes/No fields
    for field in ['HasCrCard', 'IsActiveMember']:
        custd[field] = 'Yes' if custd.get(field) == '1' else 'No'

    # 3. Generate Predictions via Loop
    results = []
    for name, model in loaded_models.items():
        try:
            pred = model.predict(new_array)[0]
            results.append({'model': name, 'prediction': decode(pred)})
        except Exception as e:
            results.append({'model': name, 'prediction': f"Error: {str(e)}"})

    return render_template('index.html', maind={'customer': custd, 'predictions': results})

if __name__ == "__main__":
    app.run(debug=True)
    
