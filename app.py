import numpy as np
import joblib
from flask import Flask, request, render_template
from tensorflow import keras

app = Flask(__name__)

# 1. Load models (keeping your original filenames)
# Note: Ensure these files exist in your 'models/' folder
try:
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'SVM (SMOTE)': joblib.load('models/svm_smote.pkl'),
        'XGBoost': joblib.load('models/xgboost.pkl'),
        'Deep Learning': keras.models.load_model('models/deep_learning.h5')
    }
except Exception as e:
    print(f"Error loading models: {e}")

def decode(prob):
    """Simple threshold check for churn"""
    return 'Will Churn' if prob > 0.5 else 'Will Stay'

@app.route('/')
def home():
    # Initial state for the table
    predictions = [{'model': name, 'prediction': ' '} for name in models.keys()]
    return render_template('index.html', maind={'customer': {}, 'predictions': predictions})

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Extract values and convert to float for math operations
    # Order must match: CreditScore, Geography, Gender, Age, Tenure, Balance, Products, HasCard, Active, Salary
    raw_values = list(request.form.values())
    numeric_values = [float(x) for x in raw_values]
    new_array = np.array(numeric_values).reshape(1, -1)

    # 3. Create customer display dictionary
    cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    custd = dict(zip(cols, raw_values))
    
    # Prettify Binary values
    for field in ['HasCrCard', 'IsActiveMember']:
        custd[field] = 'Yes' if custd.get(field) == '1' else 'No'

    # 4. Loop through models and predict
    results = []
    for name, model in models.items():
        try:
            # Handle Deep Learning (returns array) vs Scikit-Learn (returns proba)
            if name == 'Deep Learning':
                prob = float(model.predict(new_array, verbose=0)[0][0])
            else:
                # Use predict_proba for consistency if available, else predict
                prob = model.predict_proba(new_array)[0][1]
            
            results.append({
                'model': name, 
                'prediction': decode(prob),
                'probability': f"{round(prob * 100, 2)}%"
            })
        except Exception as e:
            results.append({'model': name, 'prediction': 'Error', 'probability': 'N/A'})

    # 5. Wrap in maind and return
    maind = {'customer': custd, 'predictions': results}
    return render_template('index.html', maind=maind)

if __name__ == '__main__':
    app.run(debug=True)
    
